import cv2
from ultralytics import YOLO

# Load your TensorRT engine
model = YOLO("./tensorrt_engine/yolov8n.engine", task="detect")

gst_pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=True"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open video stream. Try restarting nvargus-daemon.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference using the TensorRT engine
    results = model(frame, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("Jetson Orin YOLOv8", annotated)

    if cv2.waitKey(1) == 27: # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

