/*
$ g++ -o ds-objtracker objtrackercustom.cpp -I /opt/nvidia/deepstream/deepstream-7.1/sources/includes -I /usr/local/cuda/include $(pkg-config --cflags --libs gstreamer-1.0 glib-2.0) -L /opt/nvidia/deepstream/deepstream-7.1/lib -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvds_infer

*/
#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include <termios.h>
#include <unistd.h>
#include "gstnvdsmeta.h"
#include <fcntl.h> 
#include "nvds_analytics_meta.h"

const char* INFER_CONFIG_PATH  = "./config_infer_primary_yolo.txt";
const char* TRACKER_LIB_PATH = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so";
const char* NVDCF_CONFIG_PATH = "./config_tracker_NvDCF.yml";

static GstPadProbeReturn analytics_done_buf_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    
    for (NvDsFrameMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

        // Iterate through user metadata at the frame level for "Line Crossing" and "ROI" counts
        for (NvDsUserMetaList *l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user->data);
            
            if (user_meta->base_meta.meta_type == NVDS_USER_FRAME_META_NVDSANALYTICS) {
                NvDsAnalyticsFrameMeta *meta = (NvDsAnalyticsFrameMeta *)user_meta->user_meta_data;
                
                // Example: Print cumulative line crossing count for your "Entry" line
                if (meta->objLCCumCnt.find("Entry") != meta->objLCCumCnt.end()) {
                    std::cout << "Total Entry Count: " << meta->objLCCumCnt["Entry"] << std::endl;
                }
                
                // Current Occupancy (Who is inside ROI 'RF' right now)
                if (meta->objInROIcnt.find("RF") != meta->objInROIcnt.end()) {
                    uint32_t current_occupancy = meta->objInROIcnt["RF"];
                    std::cout << "[ROI RF] Current Occupancy: " << current_occupancy << std::endl;
                }

                // Cumulative Total (How many passed through Line 'Entry' since start)
                if (meta->objLCCumCnt.find("Entry") != meta->objLCCumCnt.end()) {
                    uint64_t total_crossed = meta->objLCCumCnt["Entry"];
                    std::cout << "[Line Entry] Total Crossed: " << total_crossed << std::endl;
                }
            }
        }
    }
    return GST_PAD_PROBE_OK;
}


static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    // Cast the 'data' pointer to the loop so we can stop it
    GMainLoop *loop = (GMainLoop *)data; 

    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop); 
            break;
        case GST_MESSAGE_ERROR: {
            gchar  *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("Error: %s\n", error->message);
            g_free(debug);
            g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

int main(int argc, char *argv[]) {
    GMainLoop *loop = NULL;
    // Removed vidconv (CPU) and nvvidconv (extra) for a cleaner CSI path
    GstElement *pipeline, *source, *capsfilter, 
               *streammux, *tracker, *analytics, *pgie, *nvvidconv2, *nvosd, *sink;
               
    GstBus *bus;
    guint bus_watch_id;

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    // logical Streammux → PGIE (YOLO) → Tracker → Analytics → Converter → OSD → Sink
    pipeline = gst_pipeline_new("yolo-pipeline");
    source = gst_element_factory_make("nvarguscamerasrc", "csi-cam-source");
    capsfilter = gst_element_factory_make("capsfilter", "caps-filter");
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    pgie = gst_element_factory_make("nvinfer", "primary-inference"); //turns raw GPU tensors into bounding boxes.
    tracker = gst_element_factory_make("nvtracker", "tracker");
    analytics = gst_element_factory_make("nvdsanalytics", "analytics");
    nvvidconv2 = gst_element_factory_make("nvvideoconvert", "nv-converter-osd");
    nvosd = gst_element_factory_make("nvdsosd", "onscreen-display");
    sink = gst_element_factory_make("nveglglessink", "egl-sink");


    if (!pipeline || !source || !capsfilter || !streammux || 
            !pgie || !tracker || !analytics || !nvvidconv2 || !nvosd || !sink) {
        std::cerr << "One or more elements could not be created." << std::endl;
        return -1;
    }

    // 1. Configure Source
    g_object_set(G_OBJECT(source), "sensor-id", 0, NULL);

    // 2. Configure Caps (Must use NVMM for CSI)
    GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1");
    g_object_set(G_OBJECT(capsfilter), "caps", caps, NULL);
    gst_caps_unref(caps);

    // 3. Configure Muxer
    g_object_set(G_OBJECT(streammux), "width", 1280, "height", 720, 
                 "batch-size", 1, "live-source", 1, NULL);
    
    // 4. Configure Inference
    g_object_set(G_OBJECT(pgie), "config-file-path", INFER_CONFIG_PATH, NULL);

    g_object_set(G_OBJECT(tracker),
    // Point to the multi-object tracker library
    "ll-lib-file", TRACKER_LIB_PATH, //tracks those boxes across frames. It is a pre-compiled library provided by NVIDIA
    // Point to your NvDCF YAML config (Template provided below)
    "ll-config-file", NVDCF_CONFIG_PATH, 
    // Scaling for tracker performance (640x384 is balanced for Orin)
    "tracker-width", 640, 
    "tracker-height", 384,
    // Enable compute-engine 0 (GPU) for Orin
    "compute-hw", 0, 
    NULL);

    g_object_set(G_OBJECT(analytics), "config-file", "config_analytics.txt", NULL);

    // Build the Pipeline
    gst_bin_add_many(GST_BIN(pipeline), source, capsfilter, 
                     streammux, pgie, tracker, analytics, nvvidconv2, nvosd, sink, NULL);

    // Link: Source -> Caps -> Muxer
    if (!gst_element_link(source, capsfilter)) {
        std::cerr << "Failed to link source to capsfilter" << std::endl;
        return -1;
    }

    GstPad *sinkpad = gst_element_request_pad_simple(streammux, "sink_0");
    GstPad *srcpad = gst_element_get_static_pad(capsfilter, "src");
    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        std::cerr << "Failed to link capsfilter to stream muxer" << std::endl;
        return -1;
    }
    gst_object_unref(sinkpad);
    gst_object_unref(srcpad);

    // Link: Muxer -> Infer ->tracker->analytics -> Conv -> OSD -> Sink
    if (!gst_element_link_many(streammux, pgie, tracker, analytics, nvvidconv2, nvosd, sink, NULL)) {
        std::cerr << "Failed to link remaining elements" << std::endl;
        return -1;
    }

    GstPad *analytics_src_pad = gst_element_get_static_pad(analytics, "src");
    if (analytics_src_pad) {
        gst_pad_add_probe(analytics_src_pad, GST_PAD_PROBE_TYPE_BUFFER, analytics_done_buf_probe, NULL, NULL);
        gst_object_unref(analytics_src_pad);
    }

    // Bus watch and keyboard listener logic remains same...
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);
    
    std::cout << "Starting Pipeline..." << std::endl;
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);

    // Cleanup...
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    return 0;
}


/* TROUBLESHOOT I2C CAMERA 

1. Hardware & Driver Verification
	Check the Node: Run v4l2-ctl --list-devices 
	Identify the Sensor: Use dmesg | grep -i imx477 to see if the kernel "bound" the driver to the I2C bus.
	The "Error -121": This is a classic "Remote I/O Error." It means the Jetson sees the sensor but can't talk to it—usually due to a loose cable, backwards ribbon, or insufficient power.
2. Physical connection
	Ribbon Orientation: On Orin Nano Super, silver pins face DOWN (toward the PCB); blue tab faces UP.
	Adapter Check: Ensure you are using the correct 15-pin to 22-pin adapter for the Orin's high-density ports.
	Power Mode: High-res sensors like the IMX477 need juice. Always set sudo nvpmodel -m 0 and sudo jetson_clocks during testing.
4. Test without Graphics (To isolate the DCE error)
	$ gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1920,height=1080' ! fakesink dump=true
	If you see a wall of hex code (text) scrolling: The camera is working! 
3. Clear the Argus Daemon
	$ sudo systemctl restart nvargus-daemon

*/

