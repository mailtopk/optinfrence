/*
  This pipeline reads an MP4 file, demuxes the H.265 stream, decodes it using hardware acceleration, runs a YOLO inference model to detect objects, draws the results on the frames, and displays the output in real-time.
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



/*	CALL BACK */
static void on_pad_added(GstElement *element, GstPad *pad, gpointer data) {
    GstCaps *caps = gst_pad_get_current_caps(pad);
    const gchar *name = gst_structure_get_name(gst_caps_get_structure(caps, 0)); 
    
    g_print("In Demux pad adding. Name: %s\n", name);
    
    // Check if the file is H265 to match our h265parse element
    if (g_str_has_prefix(name, "video/x-h265")) {
        GstElement *parser = (GstElement*)data;
        GstPad *sinkpad = gst_element_get_static_pad(parser, "sink");
        
        if (gst_pad_is_linked(sinkpad)) {
            g_print("Already linked\n");
            gst_object_unref(sinkpad);
            gst_caps_unref(caps);
            return;
        }
        
        if (gst_pad_link(pad, sinkpad) != GST_PAD_LINK_OK) {
            g_print("Failed to link demux to h265parser\n");
        } else {
            g_print("Linked demux - h265parser successfully\n");
        }
        gst_object_unref(sinkpad);
    } else {
        g_print("Skipping pad: %s (Not H265 video)\n", name);
    }
    
    gst_caps_unref(caps);
}

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
    g_print("****************** RUNNING MAIN ************");
    GMainLoop *loop = NULL;
    GstElement *pipeline, *source, *demux, *h265parser, *decoder, *nvvidconv0, *caps_filter, 
               *streammux, *pgie, *nvvidconv, *nvosd, *sink;

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    /* 1. Create Elements */
    pipeline   	= gst_pipeline_new("deepstream-file-pipeline");
    source     	= gst_element_factory_make("filesrc", "file-source");
    demux 	= gst_element_factory_make("qtdemux", "qt-demuxer"); // separates the video stream from the audio stream.
    h265parser 	= gst_element_factory_make("h265parse", "h265-parser"); // reads the stream and identifies where each frame begins and ends. "High Efficiency Video Coding" (HEVC).
    decoder    	= gst_element_factory_make("nvv4l2decoder", "nvv4l2-decoder"); 	//takes the compressed video (H.265) and turns it into raw pixel data using the Jetson’s hardware decoder.
    nvvidconv0  = gst_element_factory_make("nvvideoconvert", "nvvideo-converter0"); //Memory conversion for Orin stability
    caps_filter = gst_element_factory_make("capsfilter", "capsfilter0");  //raw pixels are in a format the AI understands (NVIDIA Memory - NVMM
    streammux  	= gst_element_factory_make("nvstreammux", "stream-muxer"); // collects video frames into "batches
    pgie       	= gst_element_factory_make("nvinfer", "primary-nvinference-engine"); // YOLO - looks at the pixels and identifies objects 
    nvvidconv  	= gst_element_factory_make("nvvideoconvert", "nvvideo-converter1"); // Converts the video so that text and boxes can be drawn on it.
    nvosd      	= gst_element_factory_make("nvdsosd", "nv-onscreendisplay"); 	//coordinates from the AI and draws the actual bounding boxes and labels on the video.
    sink       	= gst_element_factory_make("nveglglessink", "nvvideo-renderer"); // final processed video to your monitor

    if (!pipeline || !source || !demux || !h265parser || !decoder || !pgie || !sink || !nvvidconv0 || !caps_filter) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    /*Set Properties */
    g_object_set(G_OBJECT(source), "location", VIDEO_SOURCE_PATH, NULL); 
    g_object_set(G_OBJECT(streammux), "batch-size", 1, "width", 1920, "height", 1080, "batched-push-timeout", 40000, NULL);
    g_object_set(G_OBJECT(pgie), "config-file-path", INFER_CONFIG_PATH, NULL);
    

    // Force NVMM memory format for Orin
    GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
    g_object_set(G_OBJECT(caps_filter), "caps", caps, NULL);
    gst_caps_unref(caps);

    /*Add to Pipeline */
    gst_bin_add_many(GST_BIN(pipeline), source, h265parser, decoder, nvvidconv0, caps_filter, 
                     streammux, pgie, nvvidconv, nvosd, sink, demux, NULL);

    /* Logg */
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_watch(bus, bus_call, loop);
    
    /* Link Elements */
    if (!gst_element_link(source, demux)) { /*Source to Demuxer (Static link)*/
       g_printerr("Source and Demux could not be linked.\n");
       return -1;
     }

    // Connect the callback to the Demuxer, dynamic link - mp4 files
    g_signal_connect(demux, "pad-added", G_CALLBACK(on_pad_added), h265parser);

    // Link the rest of the HARDWARE chain (Static links)
    if (!gst_element_link_many(h265parser, decoder, nvvidconv0, caps_filter, NULL)) {
      g_printerr("Hardware elements could not be linked.\n");
      return -1;
    }

    // Link: CapsFilter -> Streammux (Manual Pad)
    GstPad *sinkpad = gst_element_request_pad_simple(streammux, "sink_0");
    GstPad *srcpad  = gst_element_get_static_pad(caps_filter, "src");
    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link capsfilter to streammux.\n");
        return -1;
    }

    // Link: Streammux -> PGIE -> VidConv -> OSD -> Sink
    gst_element_link_many(streammux, pgie, nvvidconv, nvosd, sink, NULL);
    
    GIOChannel *io_stdin = g_io_channel_unix_new(0); // 0 is STDIN
    g_io_add_watch(io_stdin, G_IO_IN, (GIOFunc)check_keyboard, loop);


    /*Execution */
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_print("Running...\n");
    g_main_loop_run(loop);

    /* cleanup */
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_main_loop_unref(loop);
    gst_object_unref(bus);
    gst_object_unref(sinkpad);
    gst_object_unref(srcpad);


    return 0;
}

/*
$ g++ -o ds-mp4-infer 02_ds_mp4_infrence.cpp -I /opt/nvidia/deepstream/deepstream-7.1/sources/includes -I /usr/local/cuda/include $(pkg-config --cflags --libs gstreamer-1.0 glib-2.0) -L /opt/nvidia/deepstream/deepstream-7.1/lib -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvds_infer

*/
