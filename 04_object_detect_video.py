# Using openCV

import cv2
from ultralytics import YOLO
import os

class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load your TensorRT engine
model = YOLO("./tensorrt_engine/yolov8n.engine", task="detect")


video_file = os.path.abspath("./data/video/front_4.MP4")
print(f"path is {video_file}")
gst_pipeline = (
    f"filesrc location={video_file} ! "
    "decodebin ! nvvidconv ! "
    "video/x-raw, width=640, height=360, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=True"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

window_name = "Jetson Orin - YOLO Optimized"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
cv2.resizeWindow(window_name, 640, 360) 

if not cap.isOpened():
    print("Error: Could not open video file. Check path or GStreamer support.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break

    # 3. Run Inference (stream=True saves memory)
    results = model(frame, stream=True)

    for r in results:
      r.names = class_names
      annotated_frame = r.plot()
      cv2.imshow("Object detection - Prashant", annotated_frame)

    # 4. The Exit Logic (Crucial!)
    if cv2.waitKey(1) == 27: # Press ESC to exit
        break

# 5. Clean up resources
cap.release()
cv2.destroyAllWindows()
print("Resources released successfully.")
