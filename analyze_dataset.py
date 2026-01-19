"""
데이터셋 품질 분석 스크립트
- 이미지 크기 분포
- bbox 크기 분포 (너무 작거나 큰 bbox 탐지)
- 클래스별 통계
- 잠재적 문제 탐지
"""

import os
from pathlib import Path
from collections import defaultdict
import cv2

DATASET_DIR = Path(__file__).parent / "dataset" / "unified_all"
CLASS_NAMES = ['Helmet_OFF', 'Helmet_ON', 'Vest_OFF', 'Vest_ON', 'fire', 'smoke', 'fall']


def analyze_split(split):
    """데이터 분할별 분석"""
    img_dir = DATASET_DIR / split / "images"
    lbl_dir = DATASET_DIR / split / "labels"

    if not img_dir.exists():
        return None

    stats = {
        "total_images": 0,
        "total_objects": 0,
        "class_counts": defaultdict(int),
        "img_sizes": [],
        "bbox_sizes": [],  # (width_ratio, height_ratio)
        "issues": {
            "no_label": [],
            "empty_label": [],
            "tiny_bbox": [],      # < 1% of image
            "huge_bbox": [],      # > 90% of image
            "invalid_coords": [],
        }
    }

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    stats["total_images"] = len(images)

    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        # 이미지 크기 확인 (빠른 방법)
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            stats["img_sizes"].append((w, h))

        # 라벨 파일 확인
        if not lbl_path.exists():
            stats["issues"]["no_label"].append(img_path.name)
            continue

        with open(lbl_path, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            stats["issues"]["empty_label"].append(img_path.name)
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                # 유효성 검사
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    stats["issues"]["invalid_coords"].append((img_path.name, line.strip()))
                    continue

                stats["class_counts"][cls_id] += 1
                stats["total_objects"] += 1
                stats["bbox_sizes"].append((bw, bh))

                # 크기 이상치 탐지
                area = bw * bh
                if area < 0.0001:  # 0.01% 미만
                    stats["issues"]["tiny_bbox"].append((img_path.name, cls_id, area))
                elif area > 0.81:  # 90% 이상
                    stats["issues"]["huge_bbox"].append((img_path.name, cls_id, area))

            except ValueError:
                stats["issues"]["invalid_coords"].append((img_path.name, line.strip()))

    return stats


def print_report(stats, split_name):
    """분석 결과 출력"""
    print(f"\n{'='*60}")
    print(f"[{split_name.upper()}] 데이터셋 분석")
    print(f"{'='*60}")

    print(f"\n총 이미지: {stats['total_images']:,}")
    print(f"총 객체: {stats['total_objects']:,}")
    print(f"이미지당 평균 객체: {stats['total_objects']/max(1,stats['total_images']):.2f}")

    # 클래스별 통계
    print(f"\n클래스별 객체 수:")
    for i, name in enumerate(CLASS_NAMES):
        count = stats['class_counts'].get(i, 0)
        pct = count / max(1, stats['total_objects']) * 100
        bar = '█' * int(pct / 2)
        print(f"  [{i}] {name:12s}: {count:>6,} ({pct:5.1f}%) {bar}")

    # 이미지 크기 분포
    if stats['img_sizes']:
        widths = [s[0] for s in stats['img_sizes']]
        heights = [s[1] for s in stats['img_sizes']]
        print(f"\n이미지 크기:")
        print(f"  Width:  min={min(widths)}, max={max(widths)}, avg={sum(widths)//len(widths)}")
        print(f"  Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)//len(heights)}")

    # bbox 크기 분포
    if stats['bbox_sizes']:
        areas = [w*h for w, h in stats['bbox_sizes']]
        print(f"\nBbox 크기 (이미지 대비 비율):")
        print(f"  min={min(areas)*100:.2f}%, max={max(areas)*100:.2f}%, avg={sum(areas)/len(areas)*100:.2f}%")

    # 잠재적 문제
    issues = stats['issues']
    print(f"\n⚠️  잠재적 문제:")
    print(f"  라벨 파일 없음: {len(issues['no_label'])}개")
    print(f"  빈 라벨 파일: {len(issues['empty_label'])}개")
    print(f"  너무 작은 bbox (<0.01%): {len(issues['tiny_bbox'])}개")
    print(f"  너무 큰 bbox (>81%): {len(issues['huge_bbox'])}개")
    print(f"  잘못된 좌표: {len(issues['invalid_coords'])}개")

    # 샘플 출력
    if issues['tiny_bbox'][:5]:
        print(f"\n  작은 bbox 예시:")
        for name, cls, area in issues['tiny_bbox'][:5]:
            print(f"    {name}: {CLASS_NAMES[cls]} ({area*100:.4f}%)")


def main():
    print("="*60)
    print("통합 데이터셋 품질 분석")
    print("="*60)

    for split in ["train", "valid", "test"]:
        stats = analyze_split(split)
        if stats:
            print_report(stats, split)

    print("\n" + "="*60)
    print("권장 조치:")
    print("="*60)
    print("1. tiny_bbox가 많으면 → 해당 데이터 제거 또는 확인")
    print("2. 클래스 불균형 심하면 → class weight 조정 또는 오버샘플링")
    print("3. 라벨 없는 이미지 → 제거 또는 라벨링")
    print("4. verify_labels.py로 시각적 확인 필수!")


if __name__ == "__main__":
    main()
