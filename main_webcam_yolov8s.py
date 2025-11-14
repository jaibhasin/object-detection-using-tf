import cv2
from ultralytics import YOLO

# Load YOLOv8s-OIV7 model
MODEL_PATH = (
    "models/research/object_detection/yolov8s_oiv7/yolov8s-oiv7.pt"
)
model = YOLO(MODEL_PATH)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("🎥 YOLO Webcam detection started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break

        # Run YOLO inference
        results = model(frame, verbose=False)

        # Render boxes on frame
        annotated = results[0].plot()

        cv2.imshow("YOLOv8s-OIV7 Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
