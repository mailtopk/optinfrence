/*
  This pipeline reads an MP4 file, demuxes the H.265 stream, decodes it using hardware acceleration, runs a YOLO inference model to detect objects, draws the results on the frames, and displays the output in real-time.


Debug tips
List all camera's
$ v4l2-ctl --list-devices
  
  Test both camera's are working
$ gst-launch-1.0 \
nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1920,height=1080' ! nvvidconv ! xvimagesink \
v4l2src device=/dev/video1 ! videoconvert ! xvimagesink


*/

//set clock
//sudo nvpmodel -m 0 (sets the device to maximum power mode)
//sudo jetson_clocks (locks clocks at their maximum frequency)

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include "gstnvdsmeta.h"

// Configuration Constants
const char* VIDEO_SOURCE_PATH = "../data/video/front_6.MP4";
const char* INFER_CONFIG_PATH  = "./config_infer_primary_yolo.txt";



static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    // Cast the 'data' pointer to the loop so we can stop it
    GMainLoop *loop = (GMainLoop *)data; 

    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop); // Now 'loop' is recognized!
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

/* handle escape */
static gboolean check_keyboard(GIOChannel *source, GIOCondition cond, gpointer data) {
    GMainLoop *loop = (GMainLoop *)data;
    gchar buf[1024];
    gsize bytes_read;

    if (g_io_channel_read_chars(source, buf, sizeof(buf), &bytes_read, NULL) == G_IO_STATUS_NORMAL) {
        if (buf[0] == 27) { // 27 is the Escape key
            g_print("Escape pressed. Exiting...\n");
            g_main_loop_quit(loop);
        }
    }
    return TRUE;
}
int main(int argc, char *argv[]) {
    g_print("****************** RUNNING MAIN ************\n");
    GMainLoop *loop = NULL;
    GstElement *pipeline, *source1, *source2, *streammux, *pgie, 
               *tiler, *nvvidconv, *nvosd, *sink, 
               *caps_csi, *vidconv_usb, *nvvidconv2, *caps_usb;

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    /* 1. Create Elements */
    pipeline    = gst_pipeline_new("deepstream-dual-camera-pipeline");
    
    // CSI Camera Elements
    source1     = gst_element_factory_make("nvarguscamerasrc", "csi-source");
    caps_csi    = gst_element_factory_make("capsfilter", "csi-caps");

    // USB Camera Elements
    source2     = gst_element_factory_make("v4l2src", "usb-source");
    vidconv_usb = gst_element_factory_make("videoconvert", "usb-conv");
    nvvidconv2  = gst_element_factory_make("nvvideoconvert", "usb-nvconv");
    caps_usb    = gst_element_factory_make("capsfilter", "usb-caps");

    // Core DeepStream Elements
    streammux   = gst_element_factory_make("nvstreammux", "stream-muxer");
    pgie        = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    tiler       = gst_element_factory_make("nvmultistreamtiler", "multistream-tiler");
    nvvidconv   = gst_element_factory_make("nvvideoconvert", "nvvideo-converter1");
    nvosd       = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
    sink        = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
if (!pipeline || !source1 || !source2 || !caps_csi || !vidconv_usb || !nvvidconv2 || !caps_usb || !streammux || !pgie || !tiler || !nvvidconv || !nvosd || !sink) {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
}
    if (!pipeline || !source1 || !source2 || !caps_csi || !streammux || !pgie || !tiler || !sink) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    /* 2. Set Properties */
    g_object_set(G_OBJECT(source1), "sensor-id", 0, NULL);
    g_object_set(G_OBJECT(source2), "device", "/dev/video1", NULL);
    g_object_set(G_OBJECT(pgie), "config-file-path", INFER_CONFIG_PATH, NULL);
    g_object_set(G_OBJECT(nvosd), "process-mode", 1, NULL); // 1 = GPU mode
    
    // Muxer & Tiler settings (Side-by-side 1080p)
    g_object_set(G_OBJECT(streammux), "width", 1920, "height", 1080, "batch-size", 2, "batched-push-timeout", 40000, NULL);
    g_object_set(G_OBJECT(tiler), "rows", 1, "columns", 2, "width", 1920, "height", 1080, NULL);

	/* Add this to your streammux properties */
	g_object_set(G_OBJECT(streammux), 
		     "batched-push-timeout", 40000, // 40ms (roughly 1 frame at 25fps)
		     NULL);
             
    // Caps for CSI
    GstCaps *caps1 = gst_caps_from_string("video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1");
    g_object_set(G_OBJECT(caps_csi), "caps", caps1, NULL);
    gst_caps_unref(caps1);

    // Caps for USB (Scale to match 1080p in NVMM memory)
    GstCaps *caps2 = gst_caps_from_string("video/x-raw(memory:NVMM), width=1920, height=1080");
    g_object_set(G_OBJECT(caps_usb), "caps", caps2, NULL);
    gst_caps_unref(caps2);

    /* 3. Add to Pipeline */
    gst_bin_add_many(GST_BIN(pipeline), source1, caps_csi, source2, vidconv_usb, 
                     nvvidconv2, caps_usb, streammux, pgie, tiler, nvvidconv, nvosd, sink, NULL);

    /* 4. Linking Logic */
    
    // Link CSI Branch
    if (!gst_element_link(source1, caps_csi)) return -1;
    GstPad *sinkpad0 = gst_element_request_pad_simple(streammux, "sink_0");
    GstPad *srcpad0 = gst_element_get_static_pad(caps_csi, "src");
    gst_pad_link(srcpad0, sinkpad0);
    gst_object_unref(srcpad0);

    // Link USB Branch
    if (!gst_element_link_many(source2, vidconv_usb, nvvidconv2, caps_usb, NULL)) return -1;
    GstPad *sinkpad1 = gst_element_request_pad_simple(streammux, "sink_1");
    GstPad *srcpad1 = gst_element_get_static_pad(caps_usb, "src");
    gst_pad_link(srcpad1, sinkpad1);
    gst_object_unref(srcpad1);

    // Link Batch processing line
    if (!gst_element_link_many(streammux, pgie, tiler, nvvidconv, nvosd, sink, NULL)) {
        g_printerr("Main pipeline linking failed.\n");
        return -1;
    }

    /* 5. Bus & Keyboard */
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_watch(bus, bus_call, loop);
    GIOChannel *io_stdin = g_io_channel_unix_new(0);
    g_io_add_watch(io_stdin, G_IO_IN, (GIOFunc)check_keyboard, loop);

    /* 6. Execution */
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_print("Running...\n");
    g_main_loop_run(loop);

    /* Cleanup */
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_main_loop_unref(loop);
    gst_object_unref(bus);
    return 0;
}


/*
$ g++ -o ds-dualcam-infer 03_ds_dual_cam_infrence.cpp -I /opt/nvidia/deepstream/deepstream-7.1/sources/includes -I /usr/local/cuda/include $(pkg-config --cflags --libs gstreamer-1.0 glib-2.0) -L /opt/nvidia/deepstream/deepstream-7.1/lib -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvds_infer

*/
