import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ------------------------
# MODEL PATHS
# ------------------------
MODEL_PATH = "models/research/object_detection/yolov8s_oiv7/yolov8s-oiv7.pt"
VIDEO_PATH = "output1.mp4"
OUTPUT_PATH = "output4_filtered_depth.mp4"

# Process depth estimation every N frames for speed
DEPTH_FRAME_SKIP = 20  # Process depth every 5 frames

# ------------------------
# CLASS FILTERING
# ------------------------
ALLOWED = {
    'Person', 'Man', 'Woman', 'Girl', 'Boy',
    'Car', 'Bus', 'Truck', 'Motorcycle', 'Van', 'Bicycle', 'Land vehicle',
    'Ambulance', 'Taxi',
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
# LOAD MODELS
# ------------------------
print("🔄 Loading YOLO model...")
yolo = YOLO(MODEL_PATH)

print("🔄 Loading MiDaS depth estimation model...")
device = torch.device("cpu")  # Use CPU for Mac compatibility
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device).eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

print("✅ Models loaded successfully!")

# ------------------------
# DEPTH ESTIMATION FUNCTION
# ------------------------
def estimate_depth(frame):
    """
    Estimate depth map from frame using MiDaS.
    Returns normalized depth map as numpy array.
    """
    # Prepare input - MiDaS expects RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    input_batch = transform(img_rgb).to(device)
    
    # Predict depth
    with torch.no_grad():
        prediction = midas(input_batch)
        
        # Resize to original resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convert to numpy and normalize
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return depth_map

# ------------------------
# GET AVERAGE DEPTH IN BOX
# ------------------------
def get_box_depth(depth_map, x1, y1, x2, y2):
    """
    Calculate average depth in bounding box region.
    Lower values = closer to camera.
    """
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return 0
    return np.mean(roi)

# ------------------------
# GET DISTANCE LABEL
# ------------------------
def get_distance_label(depth_value):
    """
    Convert depth value to human-readable distance.
    MiDaS outputs inverse depth - after normalization, HIGHER values = CLOSER
    """
    if depth_value > 170:
        return "VERY CLOSE", (0, 0, 255)  # Red
    elif depth_value > 128:
        return "CLOSE", (0, 165, 255)  # Orange
    elif depth_value > 85:
        return "MEDIUM", (0, 255, 255)  # Yellow
    else:
        return "FAR", (0, 255, 0)  # Green

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

    print("🎥 Processing video with YOLO + MiDaS depth estimation...")
    print(f"⚡ Processing depth every {DEPTH_FRAME_SKIP} frames for speed")
    frame_count = 0
    depth_map = None  # Cache depth map

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Estimate depth map only every N frames
        if frame_count % DEPTH_FRAME_SKIP == 1 or depth_map is None:
            depth_map = estimate_depth(frame)
        
        # Run YOLO detection
        results = yolo(frame, verbose=False)[0]
        annotated = frame.copy()

        # Store detected objects with depth for navigation
        detected_objects = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label_name = yolo.names[cls_id]

            if label_name not in ALLOWED:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Get depth information
            avg_depth = get_box_depth(depth_map, x1, y1, x2, y2)
            distance_label, color = get_distance_label(avg_depth)
            
            # Store for navigation output
            detected_objects.append({
                'label': label_name,
                'confidence': conf,
                'depth': avg_depth,
                'distance': distance_label,
                'bbox': (x1, y1, x2, y2)
            })
            
            # Create label with depth info
            label = f"{label_name} {conf:.2f} | {distance_label}"

            # Draw bounding box with color based on distance
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                annotated, 
                (x1, y1 - label_h - 10), 
                (x1 + label_w, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2
            )

        # Add frame info
        info_text = f"Frame: {frame_count} | Objects: {len(detected_objects)}"
        cv2.putText(
            annotated, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2
        )

        # Optional: Print navigation info for closest objects
        if detected_objects:
            # Sort by depth (HIGHEST values = closest now)
            detected_objects.sort(key=lambda x: x['depth'], reverse=True)
            
            # Print top 3 closest objects
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"\n📍 Frame {frame_count} - Closest objects:")
                for i, obj in enumerate(detected_objects[:3], 1):
                    print(f"  {i}. {obj['label']} - {obj['distance']} (depth: {obj['depth']:.1f})")

        # Save and show
        out.write(annotated)
        cv2.imshow("Navigation Assistant - YOLO + Depth", annotated)

        # Optional: Show depth map (grayscale visualization)
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        cv2.imshow("Depth Map", depth_colored)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Saved filtered video with depth: {OUTPUT_PATH}")
    print(f"📊 Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()