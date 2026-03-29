
# Infrence Optimization over GStream on YOLO object detection.
## Overview
This project is a real-time object detection pipeline built using NVIDIA DeepStream SDK and Triton Inference Server. It uses a YOLOv8 ONNX model to detect objects in video streams and overlay bounding boxes and class labels.
Object detection can be implemented using OpenCV with YOLO, which is simple prototypeing. However this project uses NVIDIA Deepstream for significant performance calability by using GPU Acceleration. 

The pipeline is optimized for NVIDIA GPUs, including Jetson Orin devices, and leverages TensorRT for fast inference.

Key Features:

 - Real-time object detection on video streams
 - YOLOv8 ONNX model support
 - Optimized inference using TensorRT engine
 - GPU memory management for high throughput

## Pipeline Architecture
Video processing pipeline is implemented using  DeepStream/GStreamer elements:
```
Video source File -> H264 Parser --> Decoder --> Video Convert --> Stream Multiplexing --> Primary Inference --> Video Convert --> OSD --> Renderer
```
1. Video Source
   - Reads video from file (filesrc) 
2. Decoder - Parses H.264 stream (h264parse)
   - Decodes video frames (nvv4l2decoder)
3. Video Conversion
   - Converts video frames to NVIDIA GPU memory (nvvideoconvert)
   - Applies caps filter to maintain format and resolution (capsfilter)
4. Stream Multiplexing
   - nvstreammux batches frames for inference
   - Supports multiple streams and adjustable batch size
5. Primary Inference
   - Runs YOLO model with nvinfer
   - Uses a TensorRT engine (.engine) for optimized GPU execution
   - Extracts bounding boxes and class labels using a custom YOLO parser
6. Post-Processing and Display
   - Optional conversion for overlay (nvvideoconvert)
   - Draws bounding boxes and labels (nvdsosd)
   - Renders output to screen (nveglglessink)

## Usage
### Build custom parsing lib (.so) file to interpret YOLO output
 - Bounding box extraction
 - Class score processing
 - Conversion from raw tensor output to detection objects
```bash
$ g++ -o libnvdsinfer_custom_impl_yolo.so -shared -fPIC ./nvdsinfer_custom_yolo/nvdsinfer_custom_impl_yolo.cpp \
-I /opt/nvidia/deepstream/deepstream-7.1/sources/includes/ \
-I /usr/local/cuda/include/ \
-L /opt/nvidia/deepstream/deepstream-7.1/lib/ \
-L /usr/local/cuda/lib64/ \
-L /usr/lib/aarch64-linux-gnu/ \
-lnvds_infer -lcudart
```
### Running the pipeline:
```bash
$ g++ -o my-deepstream-app basicGstreamer.cpp \
-I /opt/nvidia/deepstream/deepstream-7.1/sources/includes \
-I /usr/local/cuda/include $(pkg-config --cflags --libs gstreamer-1.0 glib-2.0) \
-L /opt/nvidia/deepstream/deepstream-7.1/lib \
-lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvds_infer
```

## Config file highlights:
- onnx-file → Path to YOLOv8 ONNX model
- model-engine-file → Path to TensorRT engine
- labelfile-path → Class labels file
- batch-size → Number of frames per inference
- process-mode → Asynchronous inference mode
- custom-lib-path & parse-bbox-func-name → Custom YOLO parser

## Inference Optimization

The project leverages DeepStream SDK and TensorRT for high-performance object detection:

1. TensorRT Engine
 - ONNX YOLOv8 model is converted into a TensorRT engine (.engine) optimized for NVIDIA GPU.
 - Optimizations applied:
     * Layer fusion
     * Kernel auto-tuning
     * Precision modes (FP32, FP16, INT8) for faster inference
2. Asynchronous and Batched Processing
  - nvstreammux can batch multiple frames to maximize GPU utilization
  - Asynchronous processing (process-mode=1) allows GPU inference while CPU handles other tasks
3. Custom YOLO Parsing
  - Custom parser library (libnvdsinfer_custom_impl_yolo.so) extracts bounding boxes and labels efficiently
  - Reduces CPU overhead during post-processing
4. GPU Memory Optimization
  - Frames converted to NVMM GPU memory to avoid CPU↔GPU transfers
  - Maintains aspect ratio to minimize unnecessary computation
