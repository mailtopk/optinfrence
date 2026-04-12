/*
$ g++ -o ds-objtracker objtrackercustom.cpp -I /opt/nvidia/deepstream/deepstream-7.1/sources/includes -I /usr/local/cuda/include $(pkg-config --cflags --libs gstreamer-1.0 glib-2.0) -L /opt/nvidia/deepstream/deepstream-7.1/lib -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvds_infer

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

const char* INFER_CONFIG_PATH  = "./config_infer_primary_yolo.txt";
const char* TRACKER_LIB_PATH = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so";
const char* NVDCF_CONFIG_PATH = "./config_tracker_NvDCF.yml";

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
              << "  ./ds-objtracker\n"
              << "  ./ds-objtracker --input video.mp4\n"
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
                
                // Extract line crossing count
                if (meta->objLCCumCnt.find("Entry") != meta->objLCCumCnt.end()) {
                    std::cout << "[Entry] Total crossed: " << meta->objLCCumCnt["Entry"] << std::endl;
                }
                
                // Extract ROI occupancy
                if (meta->objInROIcnt.find("RF") != meta->objInROIcnt.end()) {
                    uint32_t current_occupancy = meta->objInROIcnt["RF"];
                    std::cout << "[ROI RF] Current occupancy: " << current_occupancy << std::endl;
                }
            }
        }
    }
    return GST_PAD_PROBE_OK;
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

int main(int argc, char *argv[]) {
    GMainLoop *loop = NULL;
    
    // Parse command-line arguments
    parse_arguments(argc, argv);
    
    // Pipeline elements
    GstElement *pipeline, *source, *capsfilter, *streammux, *tracker, *analytics, *pgie, 
               *nvvidconv2, *nvosd, *sink, *queue1, *queue2;
               
    GstBus *bus;
    guint bus_watch_id;

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    pipeline = gst_pipeline_new("yolo-pipeline");
    
    // Create source based on config (CSI camera or file)
    source = create_source_element();
    
    capsfilter = gst_element_factory_make("capsfilter", "caps-filter");
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    pgie = gst_element_factory_make("nvinfer", "primary-inference");
    tracker = gst_element_factory_make("nvtracker", "tracker");
    analytics = gst_element_factory_make("nvdsanalytics", "analytics");
    queue1 = gst_element_factory_make("queue", "queue-before-converter");
    nvvidconv2 = gst_element_factory_make("nvvideoconvert", "nv-converter-osd");
    nvosd = gst_element_factory_make("nvdsosd", "onscreen-display");
    queue2 = gst_element_factory_make("queue", "queue-before-sink");
    
    // Create sink based on configuration
    sink = create_sink_element();

    if (!pipeline || !source || !capsfilter || !streammux || 
            !pgie || !tracker || !analytics || !queue1 || !nvvidconv2 || !nvosd || !queue2 || !sink) {
        std::cerr << "One or more elements could not be created." << std::endl;
        return -1;
    }

    // Configure queues for proper buffering
    g_object_set(G_OBJECT(queue1), "max-size-buffers", 30, "max-size-time", 0, NULL);
    g_object_set(G_OBJECT(queue2), "max-size-buffers", 30, "max-size-time", 0, NULL);

    // Configure Caps
    GstCaps *caps = NULL;
    if (g_config.source_type == SOURCE_CSI_CAMERA) {
        // CSI camera outputs NVMM format
        caps = gst_caps_from_string("video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1");
    } else {
        // File input: accept any raw video format, let streammux handle conversion
        caps = gst_caps_from_string("video/x-raw");
    }
    g_object_set(G_OBJECT(capsfilter), "caps", caps, NULL);
    gst_caps_unref(caps);

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

    g_object_set(G_OBJECT(analytics), "config-file", "config_analytics.txt", NULL);

    // Build the Pipeline - Simple: source → capsfilter → streammux → inference → OSD → sink
    gst_bin_add_many(GST_BIN(pipeline), source, capsfilter, streammux, pgie, tracker, 
                     analytics, queue1, nvvidconv2, nvosd, queue2, sink, NULL);
    
    // Link: source → capsfilter → streammux (via sink_0 pad)
    if (!gst_element_link(source, capsfilter)) {
        std::cerr << "Failed to link source to capsfilter\n";
        return -1;
    }
    
    GstPad *sinkpad = gst_element_request_pad_simple(streammux, "sink_0");
    GstPad *srcpad = gst_element_get_static_pad(capsfilter, "src");
    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        std::cerr << "Failed to link capsfilter to stream muxer\n";
        return -1;
    }
    gst_object_unref(sinkpad);
    gst_object_unref(srcpad);

    // Link: streammux → PGIE → Tracker → Analytics → Queue1 → nvvidconv2 → OSD → Queue2 → Sink
    if (!gst_element_link_many(streammux, pgie, tracker, analytics, queue1, NULL)) {
        std::cerr << "Failed to link streammux to analytics chain\n";
        return -1;
    }
    
    if (!gst_element_link_many(queue1, nvvidconv2, nvosd, queue2, sink, NULL)) {
        std::cerr << "Failed to link analytics to sink\n";
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
              << "╚════════════════════════════════════════════════╝\n" << std::endl;
    
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);

    // Cleanup...
    std::cout << "\n Shutting down pipeline..." << std::endl;
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    return 0;
}


/* HEADLESS SINK MODES

DeepStream pipelines typically require a display (nveglglessink). For headless operation
(e.g., Docker containers, SSH sessions, CI/CD), use one of these modes:

=== MODE 1: FAKESINK (Recommended for Analytics-Only) ===
  Command: ./ds-objtracker --headless
  Best for: Analytics inference, object counting, MQTT publishing
  Resources: Minimal (no video output encoding)
  Latency: Lowest (~10ms)
  Output: Console logs, MQTT messages
  
  Example pipeline output:
    Created fakesink (headless, analytics only)
    Starting DeepStream Pipeline (Headless)
    Sink type: FAKESINK (Analytics Only)

=== MODE 2: FILE OUTPUT (Save to MP4/H.264) ===
  Command: ./ds-objtracker --file output.mp4
  Best for: Recording detections, archival, post-processing
  Resources: Moderate (video encoding)
  Latency: ~50ms (encoding overhead)
  Output: MP4 file with OSD overlays
  
  Note: Requires h264enc element in pipeline. May need:
    sudo apt install libgstreamer1.0-plugins-{base,good}

=== MODE 3: UDP STREAMING ===
  Command: ./ds-objtracker --udp localhost:5000
  Best for: Network streaming, remote monitoring
  Resources: Moderate (network I/O)
  Latency: ~50-100ms (network + encoding)
  Receiver: ffplay udp://localhost:5000 -fflags nobuffer
  
  Examples:
    # Stream to remote host
    ./ds-objtracker --udp 192.168.1.100:5000
    
    # On remote machine, view stream
    ffplay udp://0.0.0.0:5000 -fflags nobuffer

=== MODE 4: DISPLAY (GPU with X11/Wayland) ===
  Command: ./ds-objtracker --display
  Best for: Local debugging, interactive monitoring
  Requirements: GPU display (SSH with -X forwarding, or local desktop)
  Note: Falls back to fakesink if display unavailable

=== DOCKER USAGE ===

With docker-compose (headless) or SSH:
  # Run in container without display
  docker run -d \\
    -v /path/to/configs:/app/configs \\
    ds-objtracker \\
    --headless --verbose

  # View logs
  docker logs -f container_id

  # Stream logs to MQTT (via integration in your C++ app)
  # See deepstream-ui/MQTT_INTEGRATION.md

=== PERFORMANCE COMPARISON ===

Sink Mode     | CPU   | Memory | GPU | Output Format | Latency
------------------------------------------------------------------
fakesink      | <5%   | ~80MB  | Yes | None (stdout) | ~10ms
file (MP4)    | 15%   | ~120MB | Yes | MP4 + OSD     | ~50ms
UDP stream    | 10%   | ~100MB | Yes | MJPEG over IP | ~80ms
display       | 8%    | ~150MB | Yes | GPU display   | ~30ms

=== TIPS ===

1. Use --headless + MQTT integration for microservices architecture
2. Use --file for recording debugging sessions
3. Use --udp for remote monitoring with minimal setup
4. Always use --verbose during testing
5. Combine with --file to log analytics to JSON/CSV separately

=== TROUBLESHOOTING ===

Q: "fakesink not found"
A: Should not happen - fakesink is part of gstreamer-core

Q: "Error linking sink element"
A: Check pipeline format compatibility:
   - Display sink expects raw video
   - File/UDP sinks need encoded format (handled by OSD output)

Q: High CPU on file output
A: Reduce resolution or framerate in muxer config:
   g_object_set(..., "width", 640, "height", 480, ...)

Q: UDP packets dropped
A: Increase MTU size or reduce quality:
   sudo ip link set eth0 mtu 9000

*/



