import shutil
from pathlib import Path

import kagglehub

MODEL_ID = "tensorflow/ssd-mobilenet-v2/tensorFlow2/ssd-mobilenet-v2"

# Updated path: Now points to object_detection folder (same level as this script)
TARGET_DIR = Path(__file__).parent / "object_detection" / "ssd_mobilenet_v2_tfhub"


def main() -> None:
    cache_path = Path(kagglehub.model_download(MODEL_ID))
    print(f"Downloaded model cache: {cache_path}")

    if (cache_path / "saved_model.pb").exists():
        saved_model_dir = cache_path
    elif (cache_path / "saved_model").exists():
        saved_model_dir = cache_path / "saved_model"
    else:
        raise FileNotFoundError(
            f"Could not locate a TensorFlow SavedModel in {cache_path}. "
            "Expected 'saved_model.pb' or a 'saved_model/' directory."
        )

    if TARGET_DIR.exists():
        print(f"Target directory already exists: {TARGET_DIR}")
    else:
        print(f"Copying SavedModel to {TARGET_DIR}")
        TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)  # Create object_detection folder if needed
        shutil.copytree(saved_model_dir, TARGET_DIR)
    print("TensorFlow SavedModel ready at:", TARGET_DIR)


if __name__ == "__main__":
    main()