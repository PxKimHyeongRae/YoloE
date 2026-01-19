"""
YOLOE Helmet/Safety Vest Detection Evaluation Script
Evaluates YOLOE model on all construction datasets for PPE detection
Saves results with bounding boxes and confidence scores
"""

import os
import cv2
import glob
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLOE
from datetime import datetime

# Base paths
BASE_PATH = Path("C:/task/yoloe")
DATASET_PATH = BASE_PATH / "dataset"
RESULTS_PATH = BASE_PATH / "results"

# Dataset configurations
DATASETS = {
    "construction_2767": {
        "path": DATASET_PATH / "construction_2767",
        "split": "test",
        "classes": {
            0: "construction vehicle",
            1: "hat",
            2: "safety wear",
            3: "worker"
        },
        "target_classes": [1, 2, 3],  # hat, safety wear, worker
        "prompts": ["hard hat", "safety vest", "construction worker"],
        "prompt_to_class": {0: 1, 1: 2, 2: 3}
    },
    "construction_3651": {
        "path": DATASET_PATH / "construction_3651",
        "split": "test",
        "classes": {
            5: "Hard Hat OFF",
            6: "Hard Hat ON",
            8: "Safety Vest OFF",
            9: "Safety Vest ON",
            14: "Worker"
        },
        "target_classes": [5, 6, 8, 9, 14],
        "prompts": [
            "person without helmet",
            "person wearing hard hat",
            "person without safety vest",
            "person wearing safety vest",
            "construction worker"
        ],
        "prompt_to_class": {0: 5, 1: 6, 2: 8, 3: 9, 4: 14}
    },
    "Construction_5458": {
        "path": DATASET_PATH / "Construction_5458",
        "split": "test",
        "classes": {
            5: "Hard Hat OFF",
            6: "Hard Hat ON",
            8: "Safety Vest OFF",
            9: "Safety Vest ON",
            14: "Worker"
        },
        "target_classes": [5, 6, 8, 9, 14],
        "prompts": [
            "person without helmet",
            "person wearing hard hat",
            "person without safety vest",
            "person wearing safety vest",
            "construction worker"
        ],
        "prompt_to_class": {0: 5, 1: 6, 2: 8, 3: 9, 4: 14}
    },
    "Construction_8845": {
        "path": DATASET_PATH / "Construction_8845",
        "split": "test",
        "classes": {
            1: "Helmet",
            8: "hat",
            9: "helmet",
            13: "no hat",
            14: "no vest",
            15: "vest"
        },
        "target_classes": [1, 8, 9, 13, 14, 15],
        "prompts": [
            "safety helmet",
            "hard hat",
            "person without hat",
            "person without vest",
            "safety vest"
        ],
        "prompt_to_class": {0: 1, 1: 8, 2: 13, 3: 14, 4: 15}
    },
    "construction_928": {
        "path": DATASET_PATH / "construction_928",
        "split": "valid",  # no test set
        "classes": {0: "class0", 1: "class1", 2: "class2"},
        "target_classes": [0, 1, 2],
        "prompts": ["hard hat", "safety vest", "person"],
        "prompt_to_class": {0: 0, 1: 1, 2: 2}
    }
}

# Colors for visualization (BGR)
COLORS = {
    "helmet_on": (0, 255, 0),      # Green
    "helmet_off": (0, 0, 255),     # Red
    "vest_on": (0, 255, 0),        # Green
    "vest_off": (0, 0, 255),       # Red
    "worker": (255, 165, 0),       # Orange
    "default": (255, 255, 0)       # Cyan
}


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [x_center, y_center, w, h] format"""
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y2 = box1[1] + box1[3] / 2

    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y2 = box2[1] + box2[3] / 2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def load_ground_truth(label_path, target_classes):
    """Load ground truth labels from YOLO format txt file"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id in target_classes:
                        labels.append({
                            'class_id': class_id,
                            'bbox': [float(parts[1]), float(parts[2]),
                                    float(parts[3]), float(parts[4])]
                        })
    return labels


def get_color_for_class(class_name):
    """Get color based on class name"""
    class_lower = class_name.lower()
    if "off" in class_lower or "without" in class_lower or "no " in class_lower:
        return (0, 0, 255)  # Red for not wearing
    elif "on" in class_lower or "wearing" in class_lower:
        return (0, 255, 0)  # Green for wearing
    elif "worker" in class_lower or "person" in class_lower:
        return (255, 165, 0)  # Orange
    else:
        return (255, 255, 0)  # Cyan default


def draw_predictions(image, predictions, class_names):
    """Draw bounding boxes and labels on image"""
    img = image.copy()
    h, w = img.shape[:2]

    for pred in predictions:
        cls_id = pred['class_id']
        conf = pred['confidence']
        bbox = pred['bbox']

        # Convert normalized coords to pixel coords
        x_center, y_center, bw, bh = bbox
        x1 = int((x_center - bw/2) * w)
        y1 = int((y_center - bh/2) * h)
        x2 = int((x_center + bw/2) * w)
        y2 = int((y_center + bh/2) * h)

        # Get class name and color
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        color = get_color_for_class(cls_name)

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label with background
        label = f"{cls_name}: {conf:.2f}"
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return img


def evaluate_dataset(model, dataset_name, config, save_images=True, max_save=50):
    """Evaluate a single dataset"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'='*60}")

    # Setup paths
    dataset_path = config["path"]
    split = config["split"]
    images_path = dataset_path / split / "images"
    labels_path = dataset_path / split / "labels"
    results_dir = RESULTS_PATH / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check if dataset exists
    if not images_path.exists():
        print(f"  WARNING: Images path not found: {images_path}")
        return None

    # Set YOLOE prompts
    prompts = config["prompts"]
    prompt_to_class = config["prompt_to_class"]
    target_classes = config["target_classes"]
    class_names = config["classes"]

    print(f"  Setting prompts: {prompts}")
    model.set_classes(prompts, model.get_text_pe(prompts))

    # Get images
    image_files = sorted(glob.glob(str(images_path / "*.jpg")))
    if not image_files:
        image_files = sorted(glob.glob(str(images_path / "*.png")))

    print(f"  Found {len(image_files)} images")

    if len(image_files) == 0:
        return None

    # Metrics
    total_tp = defaultdict(int)
    total_fp = defaultdict(int)
    total_fn = defaultdict(int)
    total_gt_count = defaultdict(int)
    total_pred_count = defaultdict(int)
    saved_count = 0

    # Process images
    for i, image_path in enumerate(image_files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} images")

        image_name = Path(image_path).stem
        label_path = labels_path / f"{image_name}.txt"

        # Load ground truth
        gt_labels = load_ground_truth(label_path, target_classes)
        for gt in gt_labels:
            total_gt_count[gt['class_id']] += 1

        # Run inference
        results = model.predict(image_path, verbose=False, conf=0.25)

        # Extract predictions
        predictions = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                img_h, img_w = result.orig_shape

                for j in range(len(boxes)):
                    xyxy = boxes.xyxy[j].cpu().numpy()
                    x_center = ((xyxy[0] + xyxy[2]) / 2) / img_w
                    y_center = ((xyxy[1] + xyxy[3]) / 2) / img_h
                    width = (xyxy[2] - xyxy[0]) / img_w
                    height = (xyxy[3] - xyxy[1]) / img_h

                    yoloe_cls = int(boxes.cls[j].cpu().numpy())
                    confidence = float(boxes.conf[j].cpu().numpy())

                    dataset_cls = prompt_to_class.get(yoloe_cls, -1)

                    if dataset_cls in target_classes:
                        predictions.append({
                            'class_id': dataset_cls,
                            'bbox': [x_center, y_center, width, height],
                            'confidence': confidence
                        })
                        total_pred_count[dataset_cls] += 1

        # Save sample images with predictions
        if save_images and saved_count < max_save and len(predictions) > 0:
            img = cv2.imread(image_path)
            if img is not None:
                result_img = draw_predictions(img, predictions, class_names)
                save_path = results_dir / f"{image_name}_pred.jpg"
                cv2.imwrite(str(save_path), result_img)
                saved_count += 1

        # Match predictions to ground truth
        matched_gt = set()
        pred_sorted = sorted(enumerate(predictions), key=lambda x: x[1]['confidence'], reverse=True)

        for pred_idx, pred in pred_sorted:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_labels):
                if gt_idx in matched_gt:
                    continue
                if pred['class_id'] != gt['class_id']:
                    continue

                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= 0.5 and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                total_tp[pred['class_id']] += 1
            else:
                total_fp[pred['class_id']] += 1

        for gt_idx, gt in enumerate(gt_labels):
            if gt_idx not in matched_gt:
                total_fn[gt['class_id']] += 1

    # Calculate metrics
    results_data = {
        "dataset": dataset_name,
        "num_images": len(image_files),
        "classes": {}
    }

    print(f"\n  {'Class':<25} {'GT':>6} {'Pred':>6} {'TP':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*80}")

    all_tp, all_fp, all_fn = 0, 0, 0

    for cls_id in target_classes:
        if cls_id not in class_names:
            continue

        cls_name = class_names[cls_id]
        tp = total_tp[cls_id]
        fp = total_fp[cls_id]
        fn = total_fn[cls_id]
        gt = total_gt_count[cls_id]
        pred = total_pred_count[cls_id]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        all_tp += tp
        all_fp += fp
        all_fn += fn

        results_data["classes"][cls_name] = {
            "gt": gt, "pred": pred, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1
        }

        print(f"  {cls_name:<25} {gt:>6} {pred:>6} {tp:>6} {precision:>8.4f} {recall:>8.4f} {f1:>8.4f}")

    # Overall
    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    results_data["overall"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1
    }

    print(f"  {'-'*80}")
    print(f"  {'OVERALL':<25} {sum(total_gt_count.values()):>6} {sum(total_pred_count.values()):>6} {all_tp:>6} {overall_precision:>8.4f} {overall_recall:>8.4f} {overall_f1:>8.4f}")
    print(f"  Saved {saved_count} sample images to {results_dir}")

    # Save results JSON
    with open(results_dir / "results.json", 'w') as f:
        json.dump(results_data, f, indent=2)

    return results_data


def main():
    """Main evaluation function"""
    print("="*60)
    print("YOLOE Helmet/Safety Vest Detection - All Datasets Evaluation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Create results directory
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading YOLOE model (yoloe-26m-seg.pt)...")
    model = YOLOE(str(BASE_PATH / "yoloe-26m-seg.pt"))
    print("Model loaded successfully!")

    # Evaluate all datasets
    all_results = {}

    for dataset_name, config in DATASETS.items():
        try:
            result = evaluate_dataset(model, dataset_name, config)
            if result:
                all_results[dataset_name] = result
        except Exception as e:
            print(f"  ERROR evaluating {dataset_name}: {e}")
            all_results[dataset_name] = {"error": str(e)}

    # Save summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE - SUMMARY")
    print("="*60)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": "yoloe-26m-seg.pt",
        "datasets": all_results
    }

    with open(RESULTS_PATH / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'Dataset':<25} {'Images':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*70)

    for name, data in all_results.items():
        if "error" in data:
            print(f"{name:<25} {'ERROR':>8}")
        elif "overall" in data:
            print(f"{name:<25} {data['num_images']:>8} {data['overall']['precision']:>10.4f} {data['overall']['recall']:>10.4f} {data['overall']['f1']:>10.4f}")

    print("-"*70)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return summary


if __name__ == "__main__":
    main()
