import cv2
from ultralytics import YOLO

# ------------------------
# MODEL PATHS
# ------------------------
MODEL_PATH = "models/research/object_detection/yolov8s_oiv7/yolov8s-oiv7.pt"
VIDEO_PATH = "v2.mp4"
OUTPUT_PATH = "output_filtered.mp4"

# ------------------------
# CLASS FILTERING
# ------------------------
ALLOWED = {
    'Person', 'Man', 'Woman', 'Girl', 'Boy',
    'Car', 'Bus', 'Truck', 'Motorcycle', 'Van', 'Bicycle', 'Land vehicle',
    'Ambulance', 'Taxi', 'Vehicle registration plate',
    'Traffic light', 'Traffic sign', 'Stop sign', 'Street light',
    'Fire hydrant', 'Parking meter',
    'Stairs', 'Door', 'Door handle', 'Building', 'House',
    'Chair', 'Table', 'Bench', 'Couch',
    'Toilet', 'Sink', 'Bed', 'Refrigerator', 'Oven', 'Microwave oven',
    'Window', 'Bathtub', 'Shower',
    'Handbag', 'Backpack', 'Suitcase', 'Luggage and bags',
    'Mobile phone', 'Bottle', 'Mug', 'Cup',
    'Wheelchair', 'Crutch', 'Stretcher',
    'Tree', 'Fountain', 'Skyscraper'
}

# ------------------------
# LOAD YOLO MODEL
# ------------------------
yolo = YOLO(MODEL_PATH)

# ------------------------
# MAIN VIDEO PIPELINE
# ------------------------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    print("🎥 Processing video with class filtering...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO
        results = yolo(frame, verbose=False)[0]
        annotated = frame.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label_name = yolo.names[cls_id]

            if label_name not in ALLOWED:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])
            label = f"{label_name} {conf:.2f}"

            # Draw box
            cv2.rectangle(
                annotated,
                (x1, y1), (x2, y2),
                (0, 255, 0),
                2
            )

            # Label text
            cv2.putText(
                annotated, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2
            )

        # Save + Show
        out.write(annotated)
        cv2.imshow("Filtered YOLO Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Saved filtered video: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
