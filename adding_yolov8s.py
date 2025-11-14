from pathlib import Path
import urllib.request

# YOLOv8-s OpenImages V7 model (600+ classes)
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-oiv7.pt"
MODEL_NAME = "yolov8s-oiv7.pt"

# Updated target directory
TARGET_DIR = (
    Path(__file__).parent
    / "models"
    / "research"
    / "object_detection"
    / "yolov8s_oiv7"
)

TARGET_MODEL_PATH = TARGET_DIR / MODEL_NAME


def download_file(url: str, output_path: Path):
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to {output_path}")


def main():
    print("Checking YOLOv8s-OIV7 model...")

    if TARGET_MODEL_PATH.exists():
        print(f"Model already exists at: {TARGET_MODEL_PATH}")
        return

    # Create directory
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Download model
    print(f"Saving model to {TARGET_MODEL_PATH}")
    download_file(MODEL_URL, TARGET_MODEL_PATH)

    print("\n🎉 YOLOv8s-OIV7 model ready!")
    print("Stored at:", TARGET_MODEL_PATH)


if __name__ == "__main__":
    main()
