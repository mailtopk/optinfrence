/*
$ sudo nvpmodel -m 0 --for MAX perf and power
$ sudo jetson_clocks

$ g++ -o ds-objtracker objtrackercustom.cpp -I /opt/nvidia/deepstream/deepstream-7.1/sources/includes -I /usr/local/cuda/include $(pkg-config --cflags --libs gstreamer-1.0 glib-2.0) -L /opt/nvidia/deepstream/deepstream-7.1/lib -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvds_infer

PIPELINE 
CSI Camera (Live) → Direct to capsfilter
MP4 File (H.265) → Demux → H.265 Parser → Hardware Decoder → Memory Convert

USAGE:
  ./ds-objtracker                         Use CSI camera  (default)
  ./ds-objtracker --input video.mp4       Analyze MP4 file (default: save to file)
  ./ds-objtracker --display               Display output on screen
  
EXAMPLES:
  ./ds-objtracker
  ./ds-objtracker --input myvideo.mp4
  ./ds-objtracker --input myvideo.mp4 --display
*/
#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include <string>
#include "gstnvdsmeta.h"
#include "nvds_analytics_meta.h"
#include <signal.h>
#include <csignal>

const char* INFER_CONFIG_PATH  = "./config_infer_primary_yolo.txt";
const char* TRACKER_LIB_PATH = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so";
const char* NVDCF_CONFIG_PATH = "./config_tracker_NvDCF.yml";
const char* ANALYTICS_CONFIG_FILE_PATH = "./config_analytics.txt";

// Global references for signal handler
static GMainLoop *g_main_loop = NULL;
static GstPipeline *g_pipeline = NULL;

// Source type enumeration
enum SourceType {
    SOURCE_CSI_CAMERA,    // CSI camera source (default)
    SOURCE_FILE           // MP4 file input
};

// Global configuration
struct PipelineConfig {
    SourceType source_type = SOURCE_CSI_CAMERA;
    std::string input_file = "";
    bool display = false;
} g_config;

void print_usage(const char* program_name) {
    std::cout << "\n=== DeepStream Object Tracker ===\n"
              << "USAGE: " << program_name << " [OPTIONS]\n\n"
              << "OPTIONS:\n"
              << "  --input <path>   Analyze MP4 file (default: CSI camera)\n"
              << "  --display        Display output on screen\n\n"
              << "EXAMPLES:\n"
              << "  ./ds-objtracker --display\n"
              << "  ./ds-objtracker --input video.mp4 --headless\n"
              << "  ./ds-objtracker --input video.mp4 --display\n\n";
}

bool parse_arguments(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        }
        else if (arg == "--input" && i + 1 < argc) {
            g_config.source_type = SOURCE_FILE;
            g_config.input_file = argv[++i];
            std::cout << "Source: FILE -> " << g_config.input_file << "\n";
        }
        else if (arg == "--display") {
            g_config.display = true;
            std::cout << "Output: Display mode\n";
        }
    }
    return true;
}

GstElement* create_source_element() {
    GstElement *source = NULL;
    
    if (g_config.source_type == SOURCE_CSI_CAMERA) {
        source = gst_element_factory_make("nvarguscamerasrc", "csi-cam-source");
        if (source) {
            g_object_set(G_OBJECT(source), "sensor-id", 0, NULL);
            std::cout << "Source: CSI Camera\n";
        }
    } else {
        source = gst_element_factory_make("filesrc", "file-source");
        if (source) {
            g_object_set(G_OBJECT(source), "location", g_config.input_file.c_str(), NULL);
            std::cout << "Source: FILE -> " << g_config.input_file << "\n";
        }
    }
    
    if (!source) {
        std::cerr << "Failed to create source element\n";
    }
    return source;
}

GstElement* create_sink_element() {
    GstElement *sink = NULL;
    
    if (g_config.display) {
        // Try to create display sink, fall back to fakesink if unavailable
        sink = gst_element_factory_make("nveglglessink", "egl-sink");
        if (!sink) {
            std::cerr << "nveglglessink not available (no display?), using fakesink\n";
            sink = gst_element_factory_make("fakesink", "sink");
            if (sink) {
                g_object_set(G_OBJECT(sink), "sync", TRUE, NULL);
            }
        } else {
            std::cout << "Output: Display (nveglglessink)\n";
        }
    } else {
        // Default: fakesink (no display)
        sink = gst_element_factory_make("fakesink", "sink");
        if (sink) {
            g_object_set(G_OBJECT(sink), "sync", TRUE, NULL);
            std::cout << "Output: Fakesink (no display)\n";
        }
    }

    if (!sink) {
        std::cerr << "Failed to create sink element\n";
    }
    return sink;
}

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
                
                // Extract cumulative line crossing count - iterates all defined line crossings
                if (!meta->objLCCumCnt.empty()) {
                    for (auto& lc : meta->objLCCumCnt) {
                        std::cout << "[Line Crossing] " << lc.first << ": " << lc.second << " total\n";
                    }
                }
                
                // Extract ROI occupancy - objects currently in defined regions
                if (!meta->objInROIcnt.empty()) {
                    for (auto& roi : meta->objInROIcnt) {
                        std::cout << "[ROI] " << roi.first << ": " << roi.second << " objects\n";
                    }
                }
            }
        }
    }
    return GST_PAD_PROBE_OK;
}

// Callback for qtdemux's pad-added signal (handles H.265 video detection)
static void on_demux_pad_added(GstElement *element, GstPad *pad, gpointer data) {
    GstCaps *caps = gst_pad_get_current_caps(pad);
    const gchar *name = gst_structure_get_name(gst_caps_get_structure(caps, 0));
    GstElement *h265parser = (GstElement *)data;
    
    std::cout << "qtdemux pad: " << name << "\n";
    
    // Check if the stream is H.265
    if (g_str_has_prefix(name, "video/x-h265")) {
        GstPad *sinkpad = gst_element_get_static_pad(h265parser, "sink");
        
        if (!gst_pad_is_linked(sinkpad)) {
            if (gst_pad_link(pad, sinkpad) == GST_PAD_LINK_OK) {
                std::cout << "qtdemux → h265parse\n";
            } else {
                std::cerr << "Failed to link qtdemux to h265parse\n";
            }
        }
        gst_object_unref(sinkpad);
    } else {
        std::cout << "Skipping pad: " << name << " (not H.265)\n";
    }
    
    gst_caps_unref(caps);
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *)data; 

    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            std::cout << "End of stream reached\n";
            g_main_loop_quit(loop); 
            break;
        case GST_MESSAGE_ERROR: {
            gchar  *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            std::cerr << "GStreamer Error: " << error->message << "\n"
                      << "   Debug: " << debug << "\n";
            g_free(debug);
            g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
        case GST_MESSAGE_WARNING: {
            gchar  *debug;
            GError *error;
            gst_message_parse_warning(msg, &error, &debug);
            std::cerr << " GStreamer Warning: " << error->message << "\n";
            g_free(debug);
            g_error_free(error);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

// Signal handler for graceful shutdown
static void signal_handler(int sig) {
    std::cout << "\n Signal " << sig << " received (Ctrl+C). Gracefully shutting down...\n";
    
    if (g_pipeline) {
        gst_element_send_event(GST_ELEMENT(g_pipeline), gst_event_new_eos());
    }
    
    if (g_main_loop) {
        g_main_loop_quit(g_main_loop);
    }
}

int main(int argc, char *argv[]) {
    
    if(argc == 1) {
        std::cout << "No command-line arguments provided.\n";
        print_usage(argv[0]);
        exit(0);
    }
    
    GMainLoop *loop = NULL;
    

    // Parse command-line arguments
    parse_arguments(argc, argv);
    
    // Pipeline elements
    GstElement *pipeline = NULL, *source = NULL, *demux = NULL, *h265parser = NULL, *decoder = NULL, *nvvidconv_decoder = NULL, *capsfilter = NULL, 
               *streammux = NULL, *tracker = NULL, *analytics = NULL, *pgie = NULL, *nvvidconv2 = NULL, *nvosd = NULL, *sink = NULL, *queue1 = NULL, *queue2 = NULL;

    GstBus *bus = NULL;
    guint bus_watch_id;

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);
    
    // Store global references for signal handler
    g_main_loop = loop;

    pipeline = gst_pipeline_new("yolo-pipeline");
    
    // Store global reference for signal handler
    g_pipeline = GST_PIPELINE(pipeline);
    
    if (g_config.source_type == SOURCE_FILE) {
        // File input requires: filesrc → qtdemux → h265parse → nvv4l2decoder
        source = gst_element_factory_make("filesrc", "file-source");
        demux = gst_element_factory_make("qtdemux", "qt-demuxer");
        h265parser = gst_element_factory_make("h265parse", "h265-parser");
        decoder = gst_element_factory_make("nvv4l2decoder", "nvv4l2-decoder");
        nvvidconv_decoder = gst_element_factory_make("nvvideoconvert", "nvvideo-converter-decoder");
        
        if (!source || !demux || !h265parser || !decoder || !nvvidconv_decoder) {
            g_printerr("Failed to create file input elements\n");
            return -1;
        }
        
        g_object_set(G_OBJECT(source), "location", g_config.input_file.c_str(), NULL);
    } else {
        // CSI camera input
        source = create_source_element();
        
        if (!source) {
            g_printerr("Failed to create CSI camera source\n");
            return -1;
        }
    }
    
    // Common elements for all sources
    capsfilter  = gst_element_factory_make("capsfilter", "caps-filter");
    streammux   = gst_element_factory_make("nvstreammux", "stream-muxer");
    pgie        = gst_element_factory_make("nvinfer", "primary-inference");
    tracker     = gst_element_factory_make("nvtracker", "tracker");
    analytics   = gst_element_factory_make("nvdsanalytics", "analytics");
    queue1      = gst_element_factory_make("queue", "queue-before-converter");
    nvvidconv2  = gst_element_factory_make("nvvideoconvert", "nv-converter-osd");
    nvosd       = gst_element_factory_make("nvdsosd", "onscreen-display");
    queue2      = gst_element_factory_make("queue", "queue-before-sink");
    sink        = create_sink_element();

    if (!pipeline || !source || !capsfilter || !streammux || 
            !pgie || !tracker || !analytics || !queue1 || !nvvidconv2 || !nvosd || !queue2 || !sink) {
        g_printerr("One or more elements could not be created\n");
        return -1;
    }

    // Configure queues for proper buffering
    g_object_set(G_OBJECT(queue1), "max-size-buffers", 30, "max-size-time", 0, NULL);
    g_object_set(G_OBJECT(queue2), "max-size-buffers", 30, "max-size-time", 0, NULL);

    // Configure Caps based on source type
    GstCaps *caps = NULL;
    if (g_config.source_type == SOURCE_CSI_CAMERA) {
        // CSI camera outputs NVMM format directly
        caps = gst_caps_from_string("video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1");
        g_object_set(G_OBJECT(capsfilter), "caps", caps, NULL);
        gst_caps_unref(caps);
    } else {
        // File input: decoder outputs raw video, capsfilter enforces NVMM format for streammux
        caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
        g_object_set(G_OBJECT(capsfilter), "caps", caps, NULL);
        gst_caps_unref(caps);
    }

    // Configure Muxer
    gboolean is_live_source = (g_config.source_type == SOURCE_CSI_CAMERA);
    g_object_set(G_OBJECT(streammux), 
                 "width", 1280, 
                 "height", 720, 
                 "batch-size", 1, 
                 "live-source", is_live_source, 
                 NULL);
    
    // Configure Inference
    g_object_set(G_OBJECT(pgie), "config-file-path", INFER_CONFIG_PATH, NULL);

    g_object_set(G_OBJECT(tracker),
        "ll-lib-file", TRACKER_LIB_PATH,
        "ll-config-file", NVDCF_CONFIG_PATH, 
        "tracker-width", 640, 
        "tracker-height", 384,
        "compute-hw", 0, 
        NULL);

    g_object_set(G_OBJECT(analytics), "config-file", ANALYTICS_CONFIG_FILE_PATH, NULL);

    // Build the Pipeline
    if (g_config.source_type == SOURCE_FILE) {
        // File: filesrc → qtdemux → h265parse → nvv4l2decoder → nvvidconv → capsfilter → streammux
        gst_bin_add_many(GST_BIN(pipeline), source, demux, h265parser, decoder, nvvidconv_decoder, 
                         capsfilter, streammux, pgie, tracker, analytics, queue1, nvvidconv2, nvosd, queue2, sink, NULL);
        
        // Link: filesrc → qtdemux (static)
        if (!gst_element_link(source, demux)) {
            g_printerr("Failed to link filesrc to qtdemux\n");
            return -1;
        }
        
        // Link: qtdemux pad-added → h265parse (dynamic callback)
        g_signal_connect(demux, "pad-added", G_CALLBACK(on_demux_pad_added), h265parser);
        
        // Link: h265parse → decoder → nvvidconv → capsfilter (static)
        if (!gst_element_link_many(h265parser, decoder, nvvidconv_decoder, capsfilter, NULL)) {
            g_printerr("Failed to link h265parse to capsfilter\n");
            return -1;
        }
    } else {
        // CSI Camera: source → capsfilter → streammux
        gst_bin_add_many(GST_BIN(pipeline), source, capsfilter, streammux, pgie, tracker, 
                         analytics, queue1, nvvidconv2, nvosd, queue2, sink, NULL);
        
        if (!gst_element_link(source, capsfilter)) {
            g_printerr("Failed to link source to capsfilter\n");
            return -1;
        }
    }
    
    // Link capsfilter to muxer (for both sources)
    GstPad *sinkpad = gst_element_request_pad_simple(streammux, "sink_0");
    GstPad *srcpad = gst_element_get_static_pad(capsfilter, "src");
    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link capsfilter to streammux\n");
        return -1;
    }
    gst_object_unref(sinkpad);
    gst_object_unref(srcpad);

    // Link: streammux → PGIE → Tracker → Analytics → Queue1 → nvvidconv2 → OSD → Queue2 → Sink
    if (!gst_element_link_many(streammux, pgie, tracker, analytics, queue1, NULL)) {
        g_printerr("Failed to link streammux to analytics\n");
        return -1;
    }
    
    if (!gst_element_link_many(queue1, nvvidconv2, nvosd, queue2, sink, NULL)) {
        g_printerr("Failed to link analytics to sink\n");
        return -1;
    }

    GstPad *analytics_src_pad = gst_element_get_static_pad(analytics, "src");
    if (analytics_src_pad) {
        gst_pad_add_probe(analytics_src_pad, GST_PAD_PROBE_TYPE_BUFFER, analytics_done_buf_probe, NULL, NULL);
        gst_object_unref(analytics_src_pad);
    }

    // Bus watch and error handling
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);
    
    // Set up signal handlers for graceful shutdown (Ctrl+C)
    signal(SIGINT, signal_handler);   // Ctrl+C
    signal(SIGTERM, signal_handler);  // Termination signal
    
    std::cout << "\n╔════════════════════════════════════════════════╗\n"
              << "║ DeepStream Object Tracker + Analytics         ║\n"
              << "║────────────────────────────────────────────────║\n"
              << "║ Source: " 
              << (g_config.source_type == SOURCE_CSI_CAMERA ? "CSI Camera (Live)" : "MP4 File")
              << "                         ║\n"
              << "║ Output: " 
              << (g_config.display ? "Display (nveglglessink)" : "Fakesink (no display)")
              << "                  ║\n"
              << "║ Model: YOLO + NvDCF Tracker                 ║\n"
              << "║ Resolution: 1280x720                         ║\n"
              << "║ Queues: Enabled (buffer management)         ║\n"
              << "║────────────────────────────────────────────────║\n"
              << "║ Press Ctrl+C to exit gracefully              ║\n"
              << "╚════════════════════════════════════════════════╝\n" << std::endl;
    
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);

    // Cleanup - graceful shutdown
    std::cout << "\nShutting down pipeline...\n";
    
    // Stop the pipeline
    gst_element_set_state(pipeline, GST_STATE_NULL);
    std::cout << "Pipeline stopped\n";
    
    // Unref the pipeline
    gst_object_unref(GST_OBJECT(pipeline));
    std::cout << " Pipeline unrefed\n";
    
    // Remove the bus watch
    g_source_remove(bus_watch_id);
    std::cout << " Bus watch removed\n";
    
    // Unref the main loop
    g_main_loop_unref(loop);
    std::cout << " Main loop unrefed\n";
    
    // Clear global references
    g_main_loop = NULL;
    g_pipeline = NULL;
    
    std::cout << " All resources cleaned up successfully\n\n";

    return 0;
}




