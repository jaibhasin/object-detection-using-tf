import cv2
from ultralytics import YOLO

# Path to your YOLO model
MODEL_PATH = "models/research/object_detection/yolov8s_oiv7/yolov8s-oiv7.pt"

# Path to your input video
VIDEO_PATH = "input.mp4"   # <-- Change this to your video file

# Output video path
OUTPUT_PATH = "output_yolo.mp4"

def main():
    # Load model
    model = YOLO(MODEL_PATH)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Could not open video: {VIDEO_PATH}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer (save output)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print("🎥 Processing video... press 'q' to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame, verbose=False)

        # Render boxes
        annotated = results[0].plot()

        # Write to output file
        out.write(annotated)

        # Show live preview
        cv2.imshow("YOLO Video Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("✅ Video saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
