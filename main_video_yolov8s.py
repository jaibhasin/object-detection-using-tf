import cv2
from ultralytics import YOLO

MODEL_PATH = "models/research/object_detection/yolov8s_oiv7/yolov8s-oiv7.pt"
VIDEO_PATH = "input.mp4"
OUTPUT_PATH = "output_filtered.mp4"

# Allowed objects
ALLOWED = {
    "Person",
    "Human face",
    "Human head",
    "Car",
    "Vehicle",
    "Bus",
    "Truck",
    "Motorcycle",
    "Vehicle registration plate",
    "License plate",
}

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print("🎥 Processing with filtered classes...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        # remove all detections not in ALLOWED list
        filtered_boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in ALLOWED:
                filtered_boxes.append(box)

        # plot filtered
        annotated = frame.copy()
        for box in filtered_boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {conf:.2f}"

            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(annotated, label, (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        out.write(annotated)
        cv2.imshow("Filtered YOLO Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("✅ Saved filtered video:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
