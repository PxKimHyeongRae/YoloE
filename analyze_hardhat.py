"""
Hard hat 데이터셋 바운딩박스 크기 분석
"""
import os
from pathlib import Path
from collections import defaultdict

dataset_path = Path(r"C:\task\yoloe\dataset\hard_hat_cleaned")

# 통계
bbox_sizes = []
class_counts = defaultdict(int)
small_objects = 0  # 5% 미만
tiny_objects = 0   # 2% 미만

for split in ["train", "valid", "test"]:
    label_dir = dataset_path / split / "labels"
    if not label_dir.exists():
        continue

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    w = float(parts[3])
                    h = float(parts[4])
                    area = w * h

                    class_counts[cls] += 1
                    bbox_sizes.append((w, h, area))

                    if area < 0.05 * 0.05:  # 5% x 5% = 0.0025
                        small_objects += 1
                    if area < 0.02 * 0.02:  # 2% x 2% = 0.0004
                        tiny_objects += 1

# 분석 결과
print("=" * 60)
print("Hard hat 데이터셋 분석")
print("=" * 60)

print(f"\n총 바운딩박스 수: {len(bbox_sizes)}")
print(f"\n클래스별 분포:")
print(f"  [0] hard hat: {class_counts[0]}")
print(f"  [1] no hat: {class_counts[1]}")

if bbox_sizes:
    widths = [b[0] for b in bbox_sizes]
    heights = [b[1] for b in bbox_sizes]
    areas = [b[2] for b in bbox_sizes]

    print(f"\n바운딩박스 크기 (상대값, 0~1):")
    print(f"  Width  - min: {min(widths):.4f}, max: {max(widths):.4f}, avg: {sum(widths)/len(widths):.4f}")
    print(f"  Height - min: {min(heights):.4f}, max: {max(heights):.4f}, avg: {sum(heights)/len(heights):.4f}")
    print(f"  Area   - min: {min(areas):.6f}, max: {max(areas):.4f}, avg: {sum(areas)/len(areas):.4f}")

    print(f"\n작은 객체 비율:")
    print(f"  Small (< 5%x5%): {small_objects} ({small_objects/len(bbox_sizes)*100:.1f}%)")
    print(f"  Tiny  (< 2%x2%): {tiny_objects} ({tiny_objects/len(bbox_sizes)*100:.1f}%)")

    # 크기 분포 히스토그램
    print(f"\n면적 분포:")
    bins = [(0, 0.001), (0.001, 0.005), (0.005, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 1.0)]
    for low, high in bins:
        count = sum(1 for a in areas if low <= a < high)
        pct = count / len(areas) * 100
        bar = "#" * int(pct / 2)
        label = f"{low:.3f}-{high:.3f}"
        print(f"  {label:12s}: {count:5d} ({pct:5.1f}%) {bar}")
