
//sudo nvpmodel -m 0 (sets the device to maximum power mode)
//sudo jetson_clocks (locks clocks at their maximum frequency)
/*
1. Hardware Memory Alignment (The SegFault Fix)
The most important change was fixing how video frames move from the Decoder to the AI Engine.
The Problem: On Orin, the nvv4l2decoder often outputs memory types that the nvstreammux can't handle directly, leading to an immediate crash.
The Solution: We added an explicit Memory Converter (nvvideoconvert) and a Caps Filter to force the video into NVMM (NVIDIA Hardware Memory) and NV12 format before it enters the muxer.
2. YOLOv8 Custom Parsing
Standard DeepStream only understands ResNet models. Since you are using YOLOv8, we had to bridge the gap:
Custom Library (.so): We updated your C++ parser code to handle the specific "transposed" output tensor of YOLOv8 (where coordinates and class scores are interleaved).
Function Mapping: We ensured the config.txt pointed to the exact function name (NvDsInferParseCustomYolo) inside that library so the AI engine knows how to draw the bounding boxes.
3. Engine Synchronization
Device Matching: We addressed the warning about "different models of devices." TensorRT engines are hardware-specific. By deleting the old .engine file, we forced the Orin to "re-build" the model specifically for its own architecture, ensuring memory buffers lined up perfectly with your parser code.
4. Pipeline Linkage
We corrected the C++ manual pad requests. In DeepStream, you cannot simply link elements to the nvstreammux; you must request a specific sink pad (like sink_0) to tell the muxer which "channel" the video belongs to.
Final Optimized Pipeline Flow:
File Source → H264 Parser → Hardware Decoder → Converter → Caps Filter (NVMM) → Streammux → nvinfer (YOLO) → OSD → EGL Renderer.
*/
#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include "gstnvdsmeta.h"

int main(int argc, char *argv[]) {
    GMainLoop *loop = NULL;
    GstElement *pipeline, *source, *h264parser, *decoder, *nvvidconv0, *caps_filter, 
               *streammux, *pgie, *nvvidconv, *nvosd, *sink;

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    /* 1. Create Elements */
    pipeline   = gst_pipeline_new("deepstream-file-pipeline");
    source     = gst_element_factory_make("filesrc", "file-source");
    h264parser = gst_element_factory_make("h264parse", "h264-parser");
    decoder    = gst_element_factory_make("nvv4l2decoder", "nvv4l2-decoder");
    
    // NEW: Memory conversion for Orin stability
    nvvidconv0  = gst_element_factory_make("nvvideoconvert", "nvvideo-converter0");
    caps_filter = gst_element_factory_make("capsfilter", "capsfilter0");
    
    streammux  = gst_element_factory_make("nvstreammux", "stream-muxer");
    pgie       = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    nvvidconv  = gst_element_factory_make("nvvideoconvert", "nvvideo-converter1");
    nvosd      = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
    sink       = gst_element_factory_make("nveglglessink", "nvvideo-renderer");

    if (!pipeline || !source || !pgie || !sink || !nvvidconv0 || !caps_filter) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    /* 2. Set Properties */
    g_object_set(G_OBJECT(source), "location", "front_4_720p_annexb.h264", NULL);
    g_object_set(G_OBJECT(streammux), "batch-size", 1, "width", 1920, "height", 1080, "batched-push-timeout", 40000, NULL);
    g_object_set(G_OBJECT(pgie), "config-file-path", "/home/ppooboni/source/optinfrence/dstream/config_infer_primary_yolo.txt", NULL);

    // Force NVMM memory format for Orin
    GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
    g_object_set(G_OBJECT(caps_filter), "caps", caps, NULL);
    gst_caps_unref(caps);

    /* 3. Add to Pipeline */
    gst_bin_add_many(GST_BIN(pipeline), source, h264parser, decoder, nvvidconv0, caps_filter, 
                     streammux, pgie, nvvidconv, nvosd, sink, NULL);

    /* 4. Link Elements */
    // Link: File -> Parser -> Decoder -> Converter -> CapsFilter
    gst_element_link_many(source, h264parser, decoder, nvvidconv0, caps_filter, NULL);

    // Link: CapsFilter -> Streammux (Manual Pad)
    GstPad *sinkpad = gst_element_get_request_pad(streammux, "sink_0");
    GstPad *srcpad  = gst_element_get_static_pad(caps_filter, "src");
    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link capsfilter to streammux.\n");
        return -1;
    }
    gst_object_unref(sinkpad);
    gst_object_unref(srcpad);

    // Link: Streammux -> PGIE -> VidConv -> OSD -> Sink
    gst_element_link_many(streammux, pgie, nvvidconv, nvosd, sink, NULL);

    /* 5. Execution */
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_print("Running...\n");
    g_main_loop_run(loop);

    /* 6. Cleanup */
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_main_loop_unref(loop);

    return 0;
}

/*
$ g++ -o my-deepstream-app basicGstreamer.cpp -I /opt/nvidia/deepstream/deepstream-7.1/sources/includes -I /usr/local/cuda/include $(pkg-config --cflags --libs gstreamer-1.0 glib-2.0) -L /opt/nvidia/deepstream/deepstream-7.1/lib -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvds_infer

*/
