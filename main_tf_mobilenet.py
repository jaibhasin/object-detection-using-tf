import sys
import tarfile
import time
import urllib.request
import os
from pathlib import Path
from typing import Dict, Iterable

# Add research directory to Python path for object_detection imports
# RESEARCH_DIR = Path(__file__).resolve().parent.parent
# if str(RESEARCH_DIR) not in sys.path:
#     sys.path.insert(0, str(RESEARCH_DIR))

import cv2
import numpy as np
import pyttsx3
import pytesseract
import tensorflow as tf
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import models as tv_models
from torchvision import transforms as trn

# Add TensorFlow Models to Python path
import sys
models_path = str(Path(__file__).parent / 'models' / 'research')
sys.path.append(models_path)
sys.path.append(str(Path(models_path) / 'slim'))

# Import TensorFlow Object Detection API
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Verify the imports work
print("Successfully imported TensorFlow Object Detection API")

# --- Global speech engine ------------------------------------------------- #
engine = pyttsx3.init()


# --- File-system paths ---------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "models" / "research" / "object_detection" / "data"


# --- TensorFlow Object Detection model configuration ---------------------- #
# Local Kaggle model path (auto-detected if exists)
KAGGLE_MODEL_DIR = ROOT_DIR / "object_detection" / "ssd_mobilenet_v2_tfhub"

# Default model name (will be downloaded if no local model found)
DEFAULT_MODEL_NAME = "ssd_mobilenet_v2_fpnlite_640x640"
MODEL_BASE_URL = os.environ.get(
    "TF_OD_MODEL_BASE",
    "http://download.tensorflow.org/models/object_detection/tf2/20200711/",
)

# Determine which model to use (env var > local Kaggle > default download)
if os.environ.get("TF_OD_MODEL"):
    # User explicitly set a model path via environment variable
    SAVED_MODEL_DIR = Path(os.environ["TF_OD_MODEL"])
    if not SAVED_MODEL_DIR.is_absolute():
        SAVED_MODEL_DIR = ROOT_DIR / SAVED_MODEL_DIR
    if (SAVED_MODEL_DIR / "saved_model.pb").exists() or (SAVED_MODEL_DIR / "saved_model").exists():
        if (SAVED_MODEL_DIR / "saved_model").exists():
            SAVED_MODEL_DIR = SAVED_MODEL_DIR / "saved_model"
    MODEL_NAME = os.environ["TF_OD_MODEL"]
elif KAGGLE_MODEL_DIR.exists() and (
    (KAGGLE_MODEL_DIR / "saved_model.pb").exists() or (KAGGLE_MODEL_DIR / "saved_model").exists()
):
    # Use local Kaggle model if it exists
    SAVED_MODEL_DIR = KAGGLE_MODEL_DIR
    if (SAVED_MODEL_DIR / "saved_model").exists():
        SAVED_MODEL_DIR = SAVED_MODEL_DIR / "saved_model"
    MODEL_NAME = "ssd_mobilenet_v2_tfhub"
else:
    # Fall back to default model (will be downloaded)
    MODEL_NAME = DEFAULT_MODEL_NAME
    MODEL_TAR = ROOT_DIR / f"{MODEL_NAME}.tar.gz"
    MODEL_DIR = ROOT_DIR / MODEL_NAME
    SAVED_MODEL_DIR = MODEL_DIR / "saved_model"

LABEL_MAP_PATH = DATA_DIR / "mscoco_label_map.pbtxt"
MIN_CONFIDENCE = float(os.environ.get("TF_OD_MIN_CONFIDENCE", 0.5))
WARNING_COOLDOWN_SECONDS = float(os.environ.get("TF_WARNING_COOLDOWN", 4.0))


# --- Places365 scene classification configuration ------------------------- #
PLACES_ARCH = "resnet18"
PLACES_MODEL_FILE = ROOT_DIR / f"whole_{PLACES_ARCH}_places365_python36.pth.tar"
PLACES_MODEL_URL = (
    f"http://places2.csail.mit.edu/models_places365/{PLACES_MODEL_FILE.name}"
)
PLACES_CATEGORIES_FILE = ROOT_DIR / "categories_places365.txt"
PLACES_CATEGORIES_URL = (
    "https://raw.githubusercontent.com/csailvision/places365/master/"
    "categories_places365.txt"
)


# --- Optional tesseract path ---------------------------------------------- #
DEFAULT_TESSERACT = Path("/opt/homebrew/bin/tesseract")
if DEFAULT_TESSERACT.exists():
    pytesseract.pytesseract.tesseract_cmd = str(DEFAULT_TESSERACT)


# --- Utility helpers ------------------------------------------------------ #
def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    print(f"Downloading {url} -> {destination}")
    with urllib.request.urlopen(url) as response, open(destination, "wb") as out_file:
        out_file.write(response.read())


def safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    if target_dir.exists() and (target_dir / "saved_model").exists():
        return
    print(f"Extracting {archive_path} to {target_dir}")
    with tarfile.open(archive_path) as tar:
        for member in tar.getmembers():
            member_path = target_dir / member.name
            if not str(member_path.resolve()).startswith(str(target_dir.resolve())):
                raise RuntimeError("Detected path traversal attempt in tar archive.")
        tar.extractall(path=ROOT_DIR)


def ensure_detection_model():
    # If using local Kaggle model or env var model, just load it
    if MODEL_NAME == "ssd_mobilenet_v2_tfhub" or os.environ.get("TF_OD_MODEL"):
        if not SAVED_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Model directory not found at {SAVED_MODEL_DIR}. "
                "Run try123.py to download the Kaggle model, or set TF_OD_MODEL env var."
            )
    else:
        # Default model: download and extract if needed
        if not SAVED_MODEL_DIR.exists():
            download_file(MODEL_BASE_URL + MODEL_TAR.name, MODEL_TAR)
            safe_extract_tar(MODEL_TAR, ROOT_DIR)
    print(f"Loading TensorFlow model from {SAVED_MODEL_DIR}")
    return tf.saved_model.load(str(SAVED_MODEL_DIR))


def ensure_category_index() -> Dict[int, Dict[str, str]]:
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(
            "Label map file missing. Expected at "
            f"{LABEL_MAP_PATH}. Did you run the TF OD API setup?"
        )
    return label_map_util.create_category_index_from_labelmap(
        str(LABEL_MAP_PATH), use_display_name=True
    )


def to_numpy(tensor_dict: Dict[str, tf.Tensor], num_detections: int) -> Dict[str, np.ndarray]:
    result = {}
    for key, value in tensor_dict.items():
        if isinstance(value, tf.Tensor):
            value = value.numpy()
        if key == "num_detections":
            result[key] = num_detections
            continue
        if isinstance(value, np.ndarray):
            if value.ndim >= 1:
                result[key] = value[0, :num_detections]
            else:
                result[key] = value
    return result


def convert_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# --- Scene classifier ----------------------------------------------------- #
class SceneClassifier:
    def __init__(self) -> None:
        download_file(PLACES_MODEL_URL, PLACES_MODEL_FILE)
        download_file(PLACES_CATEGORIES_URL, PLACES_CATEGORIES_FILE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Places365 checkpoint contains full model, needs weights_only=False
        # Safe since we're downloading from trusted MIT source
        checkpoint = torch.load(PLACES_MODEL_FILE, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model = tv_models.__dict__[PLACES_ARCH](num_classes=365)
            state_dict = {
                k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
            }
            model.load_state_dict(state_dict)
        else:
            model = checkpoint
        model.eval()
        self.model = model.to(self.device)
        self.transform = trn.Compose(
            [
                trn.Resize((256, 256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        with open(PLACES_CATEGORIES_FILE) as fh:
            self.classes = tuple(
                line.strip().split(" ")[0][3:] for line in fh if line.strip()
            )

    def describe(self, frame_bgr: np.ndarray, topk: int = 5) -> None:
        image_rgb = convert_to_rgb(frame_bgr)
        pil_image = Image.fromarray(image_rgb)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1).cpu().squeeze()
            indices = torch.argsort(probabilities, descending=True)
        engine.say("Possible scene may be")
        for idx in indices[:topk]:
            label = self.classes[idx]
            print(f"Scene suggestion: {label}")
            engine.say(label)
        engine.runAndWait()


# --- Text-to-speech helpers ---------------------------------------------- #
_last_announcements: Dict[str, float] = {}


def speak_once(key: str, message: Iterable[str]) -> None:
    now = time.time()
    last = _last_announcements.get(key, 0.0)
    if now - last < WARNING_COOLDOWN_SECONDS:
        return
    for chunk in message:
        engine.say(chunk)
    engine.runAndWait()
    _last_announcements[key] = now


# --- Vision utilities ----------------------------------------------------- #
VEHICLE_CLASS_IDS = {3, 6, 8}  # car, bus, truck
BOTTLE_CLASS_ID = 44
PERSON_CLASS_ID = 1


def estimate_distance(box: np.ndarray) -> float:
    width = box[3] - box[1]
    width = np.clip(width, 0.0, 1.0)
    return round((1.0 - width) ** 4, 1)


def handle_proximity_announcements(
    frame: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
) -> None:
    height, width, _ = frame.shape
    for idx, box in enumerate(boxes):
        score = scores[idx]
        if score < MIN_CONFIDENCE:
            continue
        class_id = int(classes[idx])
        distance = estimate_distance(box)
        mid_x = (box[1] + box[3]) / 2.0
        mid_y = (box[0] + box[2]) / 2.0
        cv2.putText(
            frame,
            f"{distance}",
            (int(mid_x * width), int(mid_y * height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        centered = 0.3 < mid_x < 0.7
        if centered and distance <= 0.5:
            if class_id in VEHICLE_CLASS_IDS:
                cv2.putText(
                    frame,
                    "WARNING!!!",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )
                speak_once(
                    "vehicle_warning",
                    ["Warning,", "vehicle approaching"],
                )
            elif class_id == BOTTLE_CLASS_ID:
                cv2.putText(
                    frame,
                    "WARNING!!!",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )
                speak_once(
                    "bottle_warning",
                    ["Warning,", "bottle very close to the frame"],
                )
            elif class_id == PERSON_CLASS_ID:
                cv2.putText(
                    frame,
                    "WARNING!!!",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )
                speak_once(
                    "person_warning",
                    ["Warning,", "person very close to the frame"],
                )
        elif class_id == BOTTLE_CLASS_ID and centered:
            speak_once(
                "bottle_safe",
                [f"Bottle at {distance} units", "Bottle is at a safer distance"],
            )
        elif class_id == PERSON_CLASS_ID and centered:
            speak_once(
                "person_safe",
                [f"Person at {distance} units", "Person is at a safer distance"],
            )


def read_text_from_frame(frame_bgr: np.ndarray) -> None:
    rgb_image = convert_to_rgb(frame_bgr)
    text = pytesseract.image_to_string(rgb_image)
    if text.strip():
        print(f"Recognized text: {text}")
        speak_once("ocr_text", [text])
    else:
        print("No text detected.")
        speak_once("ocr_none", ["I did not detect any readable text."])


def perform_detection_loop() -> None:
    detection_model = ensure_detection_model()
    category_index = ensure_category_index()
    try:
        scene_classifier = SceneClassifier()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to initialize scene classifier: {exc}")
        scene_classifier = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam. Please connect a camera device.")

    print("Starting inference loop. Press 't' or 'q' to exit, 'b' for scene, 'r' for OCR.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            # Preprocess frame for model input
            # Input shape: [1, height, width, 3] (dynamic height/width, RGB uint8)
            # The SavedModel handles resizing internally
            input_tensor = tf.convert_to_tensor(convert_to_rgb(frame))
            input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension: [H, W, 3] -> [1, H, W, 3]
            
            # Run inference
            # Outputs: detection_boxes, detection_classes, detection_scores, num_detections, etc.
            detections = detection_model(input_tensor)
            num_detections = int(detections.pop("num_detections"))
            detections = {k: v for k, v in detections.items()}
            detections_np = to_numpy(detections, num_detections)
            detection_boxes = detections_np.get("detection_boxes", np.empty((0, 4)))
            detection_scores = detections_np.get("detection_scores", np.empty(0))
            detection_classes = detections_np.get(
                "detection_classes", np.empty(0, dtype=np.int32)
            ).astype(np.int32)

            viz_utils.visualize_boxes_and_labels_on_image_array(
                frame,
                detection_boxes,
                detection_classes,
                detection_scores,
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                max_boxes_to_draw=200,
                min_score_thresh=MIN_CONFIDENCE,
            )

            handle_proximity_announcements(
                frame, detection_boxes, detection_classes, detection_scores
            )

            cv2.imshow("Assistance View", cv2.resize(frame, (1024, 768)))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("t"), ord("q")):
                print("Stopping session.")
                break
            if key == ord("b") and scene_classifier is not None:
                scene_classifier.describe(frame)
            if key == ord("r"):
                read_text_from_frame(frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    perform_detection_loop()


if __name__ == "__main__":
    main()

