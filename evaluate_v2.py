"""
YOLOE Helmet/Safety Vest Detection - Improved Version
Uses simple object prompts instead of complex state descriptions
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
RESULTS_PATH = BASE_PATH / "results_v2"

# Simple prompts for better detection
SIMPLE_PROMPTS = [
    "hard hat",           # 0 - helmet/hard hat
    "safety vest",        # 1 - safety vest/hi-vis vest
    "person",             # 2 - worker/person
]

# Colors (BGR)
COLORS = {
    0: (0, 255, 255),   # Yellow - hard hat
    1: (0, 255, 0),     # Green - safety vest
    2: (255, 165, 0),   # Orange - person
}

CLASS_NAMES = {
    0: "Hard Hat",
    1: "Safety Vest",
    2: "Person"
}

# Dataset configurations
DATASETS = {
    "construction_2767": {
        "path": DATASET_PATH / "construction_2767",
        "split": "test",
        "gt_classes": {1: "hat", 2: "safety wear", 3: "worker"},
        "mapping": {0: [1], 1: [2], 2: [3]}  # YOLOE class -> GT classes
    },
    "construction_3651": {
        "path": DATASET_PATH / "construction_3651",
        "split": "test",
        "gt_classes": {5: "Hard Hat OFF", 6: "Hard Hat ON", 8: "Safety Vest OFF", 9: "Safety Vest ON", 14: "Worker"},
        "mapping": {0: [5, 6], 1: [8, 9], 2: [14]}  # helmet -> both ON/OFF
    },
    "Construction_5458": {
        "path": DATASET_PATH / "Construction_5458",
        "split": "test",
        "gt_classes": {5: "Hard Hat OFF", 6: "Hard Hat ON", 8: "Safety Vest OFF", 9: "Safety Vest ON", 14: "Worker"},
        "mapping": {0: [5, 6], 1: [8, 9], 2: [14]}
    },
    "Construction_8845": {
        "path": DATASET_PATH / "Construction_8845",
        "split": "test",
        "gt_classes": {1: "Helmet", 8: "hat", 9: "helmet", 4: "Safety Vest", 15: "vest"},
        "mapping": {0: [1, 8, 9], 1: [4, 15], 2: [2]}  # 2 is Human
    },
}


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x_center, y_center, w, h]"""
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    inter_x1, inter_y1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    inter_x2, inter_y2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = (b1_x2-b1_x1)*(b1_y2-b1_y1) + (b2_x2-b2_x1)*(b2_y2-b2_y1) - inter_area

    return inter_area / union_area if union_area > 0 else 0


def load_gt_labels(label_path, target_classes):
    """Load ground truth from YOLO format"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in target_classes:
                        labels.append({
                            'class_id': cls_id,
                            'bbox': [float(p) for p in parts[1:5]]
                        })
    return labels


def draw_detections(image, predictions):
    """Draw bounding boxes with labels"""
    img = image.copy()
    h, w = img.shape[:2]

    for pred in predictions:
        cls_id = pred['yoloe_class']
        conf = pred['confidence']
        bbox = pred['bbox']

        # Convert to pixel coords
        x1 = int((bbox[0] - bbox[2]/2) * w)
        y1 = int((bbox[1] - bbox[3]/2) * h)
        x2 = int((bbox[0] + bbox[2]/2) * w)
        y2 = int((bbox[1] + bbox[3]/2) * h)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)

        color = COLORS.get(cls_id, (255, 255, 255))
        cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

        # Draw thick box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # Draw label background
        label = f"{cls_name}: {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Label position (above box, or inside if no space)
        ly = y1 - 10 if y1 > 30 else y1 + th + 10

        cv2.rectangle(img, (x1, ly - th - 5), (x1 + tw + 10, ly + 5), color, -1)
        cv2.putText(img, label, (x1 + 5, ly), font, font_scale, (0, 0, 0), thickness)

    return img


def evaluate_dataset(model, dataset_name, config, save_images=True, max_save=30):
    """Evaluate single dataset"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'='*60}")

    dataset_path = config["path"]
    split = config["split"]
    images_path = dataset_path / split / "images"
    labels_path = dataset_path / split / "labels"

    results_dir = RESULTS_PATH / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    if not images_path.exists():
        print(f"  ERROR: Path not found: {images_path}")
        return None

    # Get target GT classes
    gt_classes = config["gt_classes"]
    mapping = config["mapping"]
    target_gt_classes = set()
    for gt_list in mapping.values():
        target_gt_classes.update(gt_list)

    # Get images
    image_files = sorted(glob.glob(str(images_path / "*.jpg")))
    if not image_files:
        image_files = sorted(glob.glob(str(images_path / "*.png")))

    print(f"  Images: {len(image_files)}")
    print(f"  GT Classes: {gt_classes}")

    if not image_files:
        return None

    # Metrics per YOLOE class
    metrics = {i: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for i in range(3)}
    saved = 0

    for idx, img_path in enumerate(image_files):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(image_files)}")

        img_name = Path(img_path).stem
        label_path = labels_path / f"{img_name}.txt"

        # Load GT
        gt_labels = load_gt_labels(label_path, target_gt_classes)

        # Count GT per YOLOE class
        for gt in gt_labels:
            for yoloe_cls, gt_cls_list in mapping.items():
                if gt['class_id'] in gt_cls_list:
                    metrics[yoloe_cls]["gt"] += 1
                    break

        # Run inference
        results = model.predict(img_path, verbose=False, conf=0.15)  # Lower threshold

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
                    conf = float(boxes.conf[j].cpu().numpy())

                    if yoloe_cls < 3:  # Only our 3 classes
                        predictions.append({
                            'yoloe_class': yoloe_cls,
                            'bbox': [x_center, y_center, width, height],
                            'confidence': conf
                        })
                        metrics[yoloe_cls]["pred"] += 1

        # Save sample images
        if save_images and saved < max_save and len(predictions) > 0:
            img = cv2.imread(img_path)
            if img is not None:
                result_img = draw_detections(img, predictions)
                cv2.imwrite(str(results_dir / f"{img_name}_pred.jpg"), result_img)
                saved += 1

        # Match predictions to GT
        matched_gt = set()
        pred_sorted = sorted(enumerate(predictions), key=lambda x: x[1]['confidence'], reverse=True)

        for pred_idx, pred in pred_sorted:
            yoloe_cls = pred['yoloe_class']
            gt_cls_list = mapping.get(yoloe_cls, [])

            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_labels):
                if gt_idx in matched_gt:
                    continue
                if gt['class_id'] not in gt_cls_list:
                    continue

                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= 0.5 and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                metrics[yoloe_cls]["tp"] += 1
            else:
                metrics[yoloe_cls]["fp"] += 1

        # Count FN
        for gt_idx, gt in enumerate(gt_labels):
            if gt_idx not in matched_gt:
                for yoloe_cls, gt_cls_list in mapping.items():
                    if gt['class_id'] in gt_cls_list:
                        metrics[yoloe_cls]["fn"] += 1
                        break

    # Print results
    print(f"\n  {'Class':<15} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*75}")

    total = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}

    for cls_id in range(3):
        m = metrics[cls_id]
        prec = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0
        rec = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        for k in total:
            total[k] += m[k]

        print(f"  {CLASS_NAMES[cls_id]:<15} {m['gt']:>6} {m['pred']:>6} {m['tp']:>6} {m['fp']:>6} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}")

    # Overall
    prec = total["tp"] / (total["tp"] + total["fp"]) if (total["tp"] + total["fp"]) > 0 else 0
    rec = total["tp"] / (total["tp"] + total["fn"]) if (total["tp"] + total["fn"]) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"  {'-'*75}")
    print(f"  {'OVERALL':<15} {total['gt']:>6} {total['pred']:>6} {total['tp']:>6} {total['fp']:>6} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}")
    print(f"  Saved {saved} images to {results_dir}")

    return {
        "dataset": dataset_name,
        "images": len(image_files),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "metrics": {CLASS_NAMES[k]: v for k, v in metrics.items()}
    }


def main():
    print("="*60)
    print("YOLOE PPE Detection - Improved Evaluation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading YOLOE model...")
    model = YOLOE(str(BASE_PATH / "yoloe-26m-seg.pt"))

    # Set simple prompts
    print(f"Setting prompts: {SIMPLE_PROMPTS}")
    model.set_classes(SIMPLE_PROMPTS, model.get_text_pe(SIMPLE_PROMPTS))

    # Evaluate
    all_results = {}
    for name, config in DATASETS.items():
        try:
            result = evaluate_dataset(model, name, config)
            if result:
                all_results[name] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Dataset':<25} {'Images':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*65)

    for name, data in all_results.items():
        print(f"{name:<25} {data['images']:>8} {data['precision']:>10.4f} {data['recall']:>10.4f} {data['f1']:>10.4f}")

    # Save
    with open(RESULTS_PATH / "summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("-"*65)
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
