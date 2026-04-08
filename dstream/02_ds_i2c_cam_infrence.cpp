const char* INFER_CONFIG_PATH  = "./config_infer_primary_yolo.txt";

#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include <termios.h>
#include <unistd.h>
#include "gstnvdsmeta.h"
#include <fcntl.h> 


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

int main(int argc, char *argv[]) {
    GMainLoop *loop = NULL;
    // Removed vidconv (CPU) and nvvidconv (extra) for a cleaner CSI path
    GstElement *pipeline, *source, *capsfilter, 
               *streammux, *pgie, *nvvidconv2, *nvosd, *sink;
    GstBus *bus;
    guint bus_watch_id;

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    pipeline = gst_pipeline_new("yolo-pipeline");
    source = gst_element_factory_make("nvarguscamerasrc", "csi-cam-source");
    capsfilter = gst_element_factory_make("capsfilter", "caps-filter");
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    pgie = gst_element_factory_make("nvinfer", "primary-inference");
    nvvidconv2 = gst_element_factory_make("nvvideoconvert", "nv-converter-osd");
    nvosd = gst_element_factory_make("nvdsosd", "onscreen-display");
    sink = gst_element_factory_make("nveglglessink", "egl-sink");

    if (!pipeline || !source || !capsfilter || !streammux || !pgie || !sink) {
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
    g_object_set(G_OBJECT(pgie), "config-file-path", "config_infer_primary_yolo.txt", NULL);

    // Build the Pipeline
    gst_bin_add_many(GST_BIN(pipeline), source, capsfilter, 
                     streammux, pgie, nvvidconv2, nvosd, sink, NULL);

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

    // Link: Muxer -> Infer -> Conv -> OSD -> Sink
    if (!gst_element_link_many(streammux, pgie, nvvidconv2, nvosd, sink, NULL)) {
        std::cerr << "Failed to link remaining elements" << std::endl;
        return -1;
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

