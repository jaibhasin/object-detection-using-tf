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

## Models Used

### 1. YOLOv8s-OIV7 (Ultralytics)
- **Purpose**: Real-time object detection for navigation assistance
- **Dataset**: Trained on Open Images V7 (600 object classes)
- **Installation**: Run `python adding_yolov8s.py` to download the model
- **Model Location**: `models/research/object_detection/yolov8s_oiv7/yolov8s-oiv7.pt`
- **Features**: 
  - High-speed inference optimized for real-time video processing
  - Class filtering to focus on navigation-relevant objects (people, vehicles, obstacles)
  - Supports both webcam and video file input
- **Usage Scripts**: `main_webcam_yolov8s.py`, `main_video_yolov8s.py`

### 2. SSD MobileNet V2 (TensorFlow Hub)
- **Purpose**: Lightweight object detection for mobile/edge deployment
- **Architecture**: Single Shot Detector with MobileNetV2 backbone
- **Installation**: Run `python od_model_download.py` to download from Kaggle
- **Model Location**: `object_detection/ssd_mobilenet_v2_tfhub/`
- **Features**:
  - Fast inference on CPU
  - Compact model size suitable for resource-constrained devices
  - Pre-trained on COCO dataset
- **Usage Script**: `main_tf_mobilenet.py`

### 3. MiDaS (Intel ISL)
- **Purpose**: Monocular depth estimation
- **Model**: MiDaS_small variant
- **Installation**: Auto-downloads via PyTorch Hub on first run
- **Features**:
  - Estimates relative depth from single RGB images
  - Enables distance estimation without stereo cameras or sensors
  - Used for spatial awareness and proximity alerts
- **Integration**: Used in `midas_depth.py` and imported by main scripts

### 4. ResNet18-Places365 (PyTorch)
- **Purpose**: Scene classification and context understanding
- **Dataset**: Trained on Places365 (365 scene categories)
- **Installation**: Auto-downloads on first run via `main_tf_mobilenet.py`
- **Model Location**: `whole_resnet18_places365_python36.pth.tar`
- **Features**:
  - Identifies environment type (street, sidewalk, building, etc.)
  - Provides contextual awareness for better navigation
  - Activated via 'b' key during runtime

## Features

- Real-time object detection with distance estimation
- Voice alerts for nearby objects (vehicles, people, bottles)
- Scene classification using Places365
- Text recognition with OCR
- Class filtering for navigation-relevant objects

## Walkable Path Segmentation (Future Enhancement)

Instead of (or in addition to) object detection, use a **segmentation model** (like YOLOv8-Seg) trained to identify "walkable" surfaces.

### How:
You would use a model trained to classify pixels as `sidewalk`, `road`, `grass`, `curb`, etc.

### Benefit:
This allows you to give proactive guidance like:
- "Veer slightly left to stay on the sidewalk"
- "Curb detected in 2 steps"

This is far more powerful for navigation than just naming objects.
