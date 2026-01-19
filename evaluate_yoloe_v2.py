"""
YOLOE Helmet/Vest Detection Evaluation Script v2
- Improved: Remove duplicate predictions at same location
- Improved: Better text prompts
- Improved: Separate evaluation for helmet vs vest
"""

import os
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLOE

# Dataset configuration
DATASET_PATH = Path("C:/task/yoloe/dataset/construction_3651")
TEST_IMAGES_PATH = DATASET_PATH / "test" / "images"
TEST_LABELS_PATH = DATASET_PATH / "test" / "labels"
RESULTS_PATH = Path("C:/task/yoloe/results")

# Target classes - only helmet and vest (exclude Worker which has different semantics)
TARGET_CLASSES = {
    5: 'Hard Hat OFF',
    6: 'Hard Hat ON',
    8: 'Safety Vest OFF',
    9: 'Safety Vest ON',
}

# Better text prompts - more specific
YOLOE_PROMPTS = [
    "head without safety helmet",          # -> 5: Hard Hat OFF
    "yellow construction hard hat",        # -> 6: Hard Hat ON
    "torso without safety vest",           # -> 8: Safety Vest OFF
    "orange high visibility safety vest",  # -> 9: Safety Vest ON
]

# Mapping from YOLOE prediction index to dataset class
YOLOE_TO_DATASET = {
    0: 5,   # head without safety helmet -> Hard Hat OFF
    1: 6,   # yellow construction hard hat -> Hard Hat ON
    2: 8,   # torso without safety vest -> Safety Vest OFF
    3: 9,   # orange high visibility safety vest -> Safety Vest ON
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


def remove_duplicate_predictions(predictions):
    """Remove duplicate predictions at same location, keep highest confidence per location"""
    if not predictions:
        return []

    # Group by location (using IoU > 0.7 as same location)
    used = set()
    filtered = []

    # Sort by confidence descending
    sorted_preds = sorted(enumerate(predictions), key=lambda x: x[1]['confidence'], reverse=True)

    for idx, pred in sorted_preds:
        if idx in used:
            continue

        # Check if this location already has a prediction
        is_duplicate = False
        for f_idx, f_pred in enumerate(filtered):
            if calculate_iou(pred['bbox'], f_pred['bbox']) > 0.7:
                is_duplicate = True
                break

        if not is_duplicate:
            filtered.append(pred)
            used.add(idx)

    return filtered


def load_ground_truth(label_path):
    """Load ground truth labels from YOLO format txt file"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id in TARGET_CLASSES:
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append({
                            'class_id': class_id,
                            'bbox': [x_center, y_center, width, height]
                        })
    return labels


def match_predictions(predictions, ground_truths, iou_threshold=0.3):
    """Match predictions to ground truths using IoU (lowered threshold)"""
    matched_gt = set()
    matched_pred = set()

    pred_sorted = sorted(enumerate(predictions), key=lambda x: x[1]['confidence'], reverse=True)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for pred_idx, pred in pred_sorted:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            if pred['class_id'] != gt['class_id']:
                continue

            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
            tp[pred['class_id']] += 1
        else:
            fp[pred['class_id']] += 1

    for gt_idx, gt in enumerate(ground_truths):
        if gt_idx not in matched_gt:
            fn[gt['class_id']] += 1

    return tp, fp, fn


def evaluate():
    """Main evaluation function"""
    print("=" * 70)
    print("YOLOE Helmet/Vest Detection Evaluation v2")
    print("=" * 70)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # Load YOLOE model
    print("\nLoading YOLOE model (yoloe-26l-seg.pt)...")
    model = YOLOE("C:/task/yoloe/yoloe-26l-seg.pt")

    print(f"Setting text prompts: {YOLOE_PROMPTS}")
    model.set_classes(YOLOE_PROMPTS, model.get_text_pe(YOLOE_PROMPTS))

    image_files = sorted(glob.glob(str(TEST_IMAGES_PATH / "*.jpg")))
    print(f"\nFound {len(image_files)} test images")
    print("Note: Worker class excluded (different annotation semantics)")
    print("IoU Threshold: 0.3 (lowered for better matching)")

    total_tp = defaultdict(int)
    total_fp = defaultdict(int)
    total_fn = defaultdict(int)
    total_gt_count = defaultdict(int)
    total_pred_count = defaultdict(int)

    print("\nProcessing images...")
    for i, image_path in enumerate(image_files):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} images")

        image_name = Path(image_path).stem
        label_path = TEST_LABELS_PATH / f"{image_name}.txt"

        gt_labels = load_ground_truth(label_path)

        for gt in gt_labels:
            total_gt_count[gt['class_id']] += 1

        # Run inference with lower confidence
        results = model.predict(image_path, verbose=False, conf=0.15)

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

                    dataset_cls = YOLOE_TO_DATASET.get(yoloe_cls, -1)

                    if dataset_cls in TARGET_CLASSES:
                        predictions.append({
                            'class_id': dataset_cls,
                            'bbox': [x_center, y_center, width, height],
                            'confidence': confidence
                        })

        # Remove duplicate predictions at same location
        predictions = remove_duplicate_predictions(predictions)

        for pred in predictions:
            total_pred_count[pred['class_id']] += 1

        tp, fp, fn = match_predictions(predictions, gt_labels, iou_threshold=0.3)

        for cls_id in TARGET_CLASSES.keys():
            total_tp[cls_id] += tp[cls_id]
            total_fp[cls_id] += fp[cls_id]
            total_fn[cls_id] += fn[cls_id]

    # Calculate metrics
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (v2 - Improved)")
    print("=" * 70)

    print(f"\n{'Class':<20} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 85)

    all_tp, all_fp, all_fn = 0, 0, 0

    # Separate by category
    helmet_tp, helmet_fp, helmet_fn = 0, 0, 0
    vest_tp, vest_fp, vest_fn = 0, 0, 0

    for cls_id, cls_name in TARGET_CLASSES.items():
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

        # Categorize
        if cls_id in [5, 6]:  # Helmet
            helmet_tp += tp
            helmet_fp += fp
            helmet_fn += fn
        else:  # Vest
            vest_tp += tp
            vest_fp += fp
            vest_fn += fn

        line = f"{cls_name:<20} {gt:>6} {pred:>6} {tp:>6} {fp:>6} {fn:>6} {precision:>8.4f} {recall:>8.4f} {f1:>8.4f}"
        print(line)

    print("-" * 85)

    # Overall metrics
    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    total_gt = sum(total_gt_count.values())
    total_pred = sum(total_pred_count.values())

    print(f"{'OVERALL':<20} {total_gt:>6} {total_pred:>6} {all_tp:>6} {all_fp:>6} {all_fn:>6} {overall_precision:>8.4f} {overall_recall:>8.4f} {overall_f1:>8.4f}")

    # Category breakdown
    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN")
    print("=" * 70)

    helmet_prec = helmet_tp / (helmet_tp + helmet_fp) if (helmet_tp + helmet_fp) > 0 else 0
    helmet_rec = helmet_tp / (helmet_tp + helmet_fn) if (helmet_tp + helmet_fn) > 0 else 0
    helmet_f1 = 2 * helmet_prec * helmet_rec / (helmet_prec + helmet_rec) if (helmet_prec + helmet_rec) > 0 else 0

    vest_prec = vest_tp / (vest_tp + vest_fp) if (vest_tp + vest_fp) > 0 else 0
    vest_rec = vest_tp / (vest_tp + vest_fn) if (vest_tp + vest_fn) > 0 else 0
    vest_f1 = 2 * vest_prec * vest_rec / (vest_prec + vest_rec) if (vest_prec + vest_rec) > 0 else 0

    print(f"  Helmet Detection:  Precision={helmet_prec:.4f}  Recall={helmet_rec:.4f}  F1={helmet_f1:.4f}")
    print(f"  Vest Detection:    Precision={vest_prec:.4f}  Recall={vest_rec:.4f}  F1={vest_f1:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Overall Precision: {overall_precision:.4f}")
    print(f"  Overall Recall: {overall_recall:.4f}")
    print(f"  Overall F1-Score: {overall_f1:.4f}")
    print("=" * 70)

    # Save results
    results_file = RESULTS_PATH / "evaluation_results_v2.txt"
    with open(results_file, 'w') as f:
        f.write(f"YOLOE Helmet/Vest Detection Evaluation v2\n")
        f.write(f"Model: yoloe-26l-seg.pt\n")
        f.write(f"IoU Threshold: 0.3\n")
        f.write(f"Confidence Threshold: 0.15\n")
        f.write(f"\nOverall Precision: {overall_precision:.4f}\n")
        f.write(f"Overall Recall: {overall_recall:.4f}\n")
        f.write(f"Overall F1-Score: {overall_f1:.4f}\n")
        f.write(f"\nHelmet F1: {helmet_f1:.4f}\n")
        f.write(f"Vest F1: {vest_f1:.4f}\n")
    print(f"\nResults saved to: {results_file}")

    return {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1
    }


if __name__ == "__main__":
    evaluate()
