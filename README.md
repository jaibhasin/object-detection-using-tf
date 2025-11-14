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

3. Run the application:
```bash
python main.py
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
