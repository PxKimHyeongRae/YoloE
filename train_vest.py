"""
Safety Vest Detection Training Script (YOLOE)
Dataset: ppe_vest_13k (13,640 train images)
Classes: no_safety_vest, safety_vest
"""

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer
from pathlib import Path

# Paths
BASE_PATH = Path(__file__).parent
MODEL_YAML = "yoloe-26n.yaml"  # YOLOE architecture config
MODEL_WEIGHTS = BASE_PATH / "models" / "yoloe-26n-seg.pt"  # Pretrained weights
DATA_PATH = BASE_PATH / "dataset" / "ppe_vest_13k" / "data.yaml"

# Training config
CONFIG = {
    "data": str(DATA_PATH),
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "name": "vest_detection",
    "project": str(BASE_PATH / "runs" / "detect"),
    "patience": 20,
    "device": 0,
    "trainer": YOLOEPETrainer,  # YOLOE Detection Trainer
}

if __name__ == "__main__":
    print("=" * 50)
    print("YOLOE Safety Vest Detection Training")
    print("=" * 50)
    print(f"Model YAML: {MODEL_YAML}")
    print(f"Weights: {MODEL_WEIGHTS}")
    print(f"Data: {DATA_PATH}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch: {CONFIG['batch']}")
    print("=" * 50)

    # Initialize model from YAML config
    model = YOLOE(MODEL_YAML)

    # Load pretrained weights
    model.load(str(MODEL_WEIGHTS))

    # Train with YOLOEPETrainer
    results = model.train(**CONFIG)

    print("\nTraining complete!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
