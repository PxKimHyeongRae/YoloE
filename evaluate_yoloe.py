"""
YOLOE Helmet/Vest Detection Evaluation Script
Evaluates YOLOE model on construction_3651 dataset for PPE detection
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

# Dataset class mapping (from data.yaml)
DATASET_CLASSES = {
    0: 'Dump Truck',
    1: 'Excavator',
    2: 'Front End Loader',
    3: 'Gloves ON',
    4: 'Gloves-OFF',
    5: 'Hard Hat OFF',
    6: 'Hard Hat ON',
    7: 'Ladder',
    8: 'Safety Vest OFF',
    9: 'Safety Vest ON',
    10: 'Skid Steer',
    11: 'Tractor Trailer',
    12: 'Trailer',
    13: 'Vehicle',
    14: 'Worker'
}

# Target classes for helmet/vest detection
TARGET_CLASSES = {
    5: 'Hard Hat OFF',
    6: 'Hard Hat ON',
    8: 'Safety Vest OFF',
    9: 'Safety Vest ON',
    14: 'Worker'
}

# YOLOE text prompts (mapped to dataset classes) - improved prompts
YOLOE_PROMPTS = [
    "person not wearing helmet",          # -> 5: Hard Hat OFF
    "person wearing yellow hard hat",     # -> 6: Hard Hat ON
    "person without reflective vest",     # -> 8: Safety Vest OFF
    "person wearing orange safety vest",  # -> 9: Safety Vest ON
    "worker"                              # -> 14: Worker
]

# Mapping from YOLOE prediction index to dataset class
YOLOE_TO_DATASET = {
    0: 5,   # person without hard hat -> Hard Hat OFF
    1: 6,   # person wearing hard hat -> Hard Hat ON
    2: 8,   # person without safety vest -> Safety Vest OFF
    3: 9,   # person wearing safety vest -> Safety Vest ON
    4: 14   # construction worker -> Worker
}

# Reverse mapping
DATASET_TO_YOLOE = {v: k for k, v in YOLOE_TO_DATASET.items()}


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [x_center, y_center, w, h] format"""
    # Convert to [x1, y1, x2, y2]
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y2 = box1[1] + box1[3] / 2

    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y2 = box2[1] + box2[3] / 2

    # Intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def load_ground_truth(label_path):
    """Load ground truth labels from YOLO format txt file"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    # Only keep target classes
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


def match_predictions(predictions, ground_truths, iou_threshold=0.5):
    """Match predictions to ground truths using IoU"""
    matched_gt = set()
    matched_pred = set()

    # Sort predictions by confidence (descending)
    pred_sorted = sorted(enumerate(predictions), key=lambda x: x[1]['confidence'], reverse=True)

    tp = defaultdict(int)  # True Positives per class
    fp = defaultdict(int)  # False Positives per class
    fn = defaultdict(int)  # False Negatives per class

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

    # Count false negatives (unmatched ground truths)
    for gt_idx, gt in enumerate(ground_truths):
        if gt_idx not in matched_gt:
            fn[gt['class_id']] += 1

    return tp, fp, fn


def evaluate():
    """Main evaluation function"""
    print("=" * 60)
    print("YOLOE Helmet/Vest Detection Evaluation")
    print("=" * 60)

    # Create results directory
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # Load YOLOE model - using larger model for better accuracy
    print("\nLoading YOLOE model (yoloe-26l-seg.pt)...")
    model = YOLOE("C:/task/yoloe/yoloe-26l-seg.pt")

    # Set text prompts for target classes
    print(f"Setting text prompts: {YOLOE_PROMPTS}")
    model.set_classes(YOLOE_PROMPTS, model.get_text_pe(YOLOE_PROMPTS))

    # Get test images
    image_files = sorted(glob.glob(str(TEST_IMAGES_PATH / "*.jpg")))
    print(f"\nFound {len(image_files)} test images")

    # Initialize metrics
    total_tp = defaultdict(int)
    total_fp = defaultdict(int)
    total_fn = defaultdict(int)
    total_gt_count = defaultdict(int)
    total_pred_count = defaultdict(int)

    # Process each image
    print("\nProcessing images...")
    for i, image_path in enumerate(image_files):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} images")

        # Get corresponding label file
        image_name = Path(image_path).stem
        label_path = TEST_LABELS_PATH / f"{image_name}.txt"

        # Load ground truth
        gt_labels = load_ground_truth(label_path)

        # Count ground truth per class
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
                    # Get box in xyxy format and convert to normalized xywh
                    xyxy = boxes.xyxy[j].cpu().numpy()
                    x_center = ((xyxy[0] + xyxy[2]) / 2) / img_w
                    y_center = ((xyxy[1] + xyxy[3]) / 2) / img_h
                    width = (xyxy[2] - xyxy[0]) / img_w
                    height = (xyxy[3] - xyxy[1]) / img_h

                    yoloe_cls = int(boxes.cls[j].cpu().numpy())
                    confidence = float(boxes.conf[j].cpu().numpy())

                    # Map YOLOE class to dataset class
                    dataset_cls = YOLOE_TO_DATASET.get(yoloe_cls, -1)

                    if dataset_cls in TARGET_CLASSES:
                        predictions.append({
                            'class_id': dataset_cls,
                            'bbox': [x_center, y_center, width, height],
                            'confidence': confidence
                        })
                        total_pred_count[dataset_cls] += 1

        # Match predictions to ground truth
        tp, fp, fn = match_predictions(predictions, gt_labels)

        for cls_id in TARGET_CLASSES.keys():
            total_tp[cls_id] += tp[cls_id]
            total_fp[cls_id] += fp[cls_id]
            total_fn[cls_id] += fn[cls_id]

    # Calculate metrics per class
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    results_text = []
    results_text.append("YOLOE Helmet/Vest Detection Evaluation Results")
    results_text.append("=" * 60)
    results_text.append(f"Model: yoloe-26l-seg.pt")
    results_text.append(f"Dataset: construction_3651")
    results_text.append(f"Test Images: {len(image_files)}")
    results_text.append(f"IoU Threshold: 0.5")
    results_text.append("")

    print(f"\n{'Class':<25} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 90)
    results_text.append(f"{'Class':<25} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    results_text.append("-" * 90)

    all_tp, all_fp, all_fn = 0, 0, 0
    class_aps = []

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

        if gt > 0:
            class_aps.append(precision * recall)

        line = f"{cls_name:<25} {gt:>6} {pred:>6} {tp:>6} {fp:>6} {fn:>6} {precision:>8.4f} {recall:>8.4f} {f1:>8.4f}"
        print(line)
        results_text.append(line)

    print("-" * 90)
    results_text.append("-" * 90)

    # Overall metrics
    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    mean_ap = np.mean(class_aps) if class_aps else 0

    total_gt = sum(total_gt_count.values())
    total_pred = sum(total_pred_count.values())

    line = f"{'OVERALL':<25} {total_gt:>6} {total_pred:>6} {all_tp:>6} {all_fp:>6} {all_fn:>6} {overall_precision:>8.4f} {overall_recall:>8.4f} {overall_f1:>8.4f}"
    print(line)
    results_text.append(line)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Mean Average Precision (mAP@0.5): {mean_ap:.4f}")
    print(f"  Overall Precision: {overall_precision:.4f}")
    print(f"  Overall Recall: {overall_recall:.4f}")
    print(f"  Overall F1-Score: {overall_f1:.4f}")
    print("=" * 60)

    results_text.append("")
    results_text.append("=" * 60)
    results_text.append("SUMMARY")
    results_text.append("=" * 60)
    results_text.append(f"  Mean Average Precision (mAP@0.5): {mean_ap:.4f}")
    results_text.append(f"  Overall Precision: {overall_precision:.4f}")
    results_text.append(f"  Overall Recall: {overall_recall:.4f}")
    results_text.append(f"  Overall F1-Score: {overall_f1:.4f}")
    results_text.append("=" * 60)

    # Save results
    results_file = RESULTS_PATH / "evaluation_results_26l.txt"
    with open(results_file, 'w') as f:
        f.write('\n'.join(results_text))
    print(f"\nResults saved to: {results_file}")

    return {
        'mAP': mean_ap,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1
    }


if __name__ == "__main__":
    evaluate()
