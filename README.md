# Object Detection Assistance System

A real-time object detection and scene classification system using TensorFlow and PyTorch for assistive navigation.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the object detection model:
```bash
python od_model_download.py
```

3. Download YOLOv8s-OIV7 model:
```bash
python adding_yolov8s.py
```

4. Run the application:
```bash
# TensorFlow model with webcam
python main.py

# YOLOv8s model with webcam
python main_webcam_yolov8s.py

# YOLOv8s model with video file
python main_video_yolov8s.py
```

## Usage

- Press **'q'** or **'t'** to quit
- Press **'b'** for scene classification
- Press **'r'** for OCR text recognition

## Features

- Real-time object detection with distance estimation
- Voice alerts for nearby objects (vehicles, people, bottles)
- Scene classification using Places365
- Text recognition with OCR
- **YOLOv8s-OIV7** with class filtering for navigation-relevant objects

## Walkable Path Segmentation (Future Enhancement)

Instead of (or in addition to) object detection, use a **segmentation model** (like YOLOv8-Seg) trained to identify "walkable" surfaces.

### How:
You would use a model trained to classify pixels as `sidewalk`, `road`, `grass`, `curb`, etc.

### Benefit:
This allows you to give proactive guidance like:
- "Veer slightly left to stay on the sidewalk"
- "Curb detected in 2 steps"

This is far more powerful for navigation than just naming objects.
