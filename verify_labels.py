"""
라벨 시각적 검증 스크립트
- 랜덤 이미지를 뽑아서 bbox와 함께 표시
- 라벨이 정확한지 육안으로 확인
"""

import cv2
import random
from pathlib import Path

# 설정
DATASET_DIR = Path(__file__).parent / "dataset" / "unified_all"
CLASS_NAMES = ['Helmet_OFF', 'Helmet_ON', 'Vest_OFF', 'Vest_ON', 'fire', 'smoke', 'fall']
COLORS = [
    (0, 0, 255),    # Helmet_OFF - 빨강
    (0, 255, 0),    # Helmet_ON - 초록
    (0, 165, 255),  # Vest_OFF - 주황
    (255, 255, 0),  # Vest_ON - 청록
    (0, 0, 255),    # fire - 빨강
    (128, 128, 128),# smoke - 회색
    (255, 0, 255),  # fall - 마젠타
]


def draw_boxes(img_path, label_path):
    """이미지에 bbox 그리기"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"이미지 로드 실패: {img_path}")
        return None

    h, w = img.shape[:2]

    if not label_path.exists():
        return img

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            # YOLO format → pixel coordinates
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)

            color = COLORS[cls_id] if cls_id < len(COLORS) else (255, 255, 255)
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


def verify_samples(split="train", num_samples=20):
    """랜덤 샘플 검증"""
    img_dir = DATASET_DIR / split / "images"
    lbl_dir = DATASET_DIR / split / "labels"

    if not img_dir.exists():
        print(f"디렉토리 없음: {img_dir}")
        return

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    samples = random.sample(images, min(num_samples, len(images)))

    output_dir = Path(__file__).parent / "label_verification"
    output_dir.mkdir(exist_ok=True)

    print(f"\n{split} 데이터셋에서 {len(samples)}개 샘플 검증 중...")
    print(f"결과 저장: {output_dir}")

    for i, img_path in enumerate(samples):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        result = draw_boxes(img_path, lbl_path)

        if result is not None:
            out_path = output_dir / f"{split}_{i:03d}_{img_path.name}"
            cv2.imwrite(str(out_path), result)
            print(f"  [{i+1}/{len(samples)}] {img_path.name}")

    print(f"\n완료! {output_dir} 폴더에서 이미지를 확인하세요.")


if __name__ == "__main__":
    print("="*60)
    print("라벨 검증 - 랜덤 샘플 시각화")
    print("="*60)

    verify_samples("train", 30)
    verify_samples("valid", 10)

    print("\n" + "="*60)
    print("확인 사항:")
    print("1. bbox가 객체를 정확히 감싸는지")
    print("2. 클래스 라벨이 올바른지")
    print("3. 누락된 객체가 없는지")
    print("="*60)
