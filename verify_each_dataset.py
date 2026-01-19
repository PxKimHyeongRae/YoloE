"""
각 원본 데이터셋별 라벨 품질 검증
- 데이터셋별로 50장씩 샘플링
- 바운딩박스 시각화하여 품질 판단
"""

import cv2
import random
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent / "dataset"
OUTPUT_DIR = Path(__file__).parent / "dataset_quality_check"

# 검사할 데이터셋 목록과 클래스 정보
DATASETS = {
    # PPE 데이터셋
    "construction safety.v2-release.yolo26_helmet_no-helmet_no-vest_vest_person": {
        "classes": ["helmet", "no-helmet", "no-vest", "person", "vest"],
        "category": "PPE"
    },
    "ppe_only_700_Hard_Hat_OFF', 'Hard_Hat_ON', 'Safety_Vest_OFF', 'Safety_Vest_ON'": {
        "classes": ["Hard_Hat_OFF", "Hard_Hat_ON", "Safety_Vest_OFF", "Safety_Vest_ON"],
        "category": "PPE"
    },
    "ppe_vest_13k": {
        "classes": ["no_safety_vest", "safety_vest"],
        "category": "PPE"
    },
    "Safety vest.v3i.yolo26_vest_no-vest_6k": {
        "classes": ["NO-Safety Vest", "Safety Vest"],
        "category": "PPE"
    },
    "PPE-0.3.v1i.yolo26": {
        "classes": None,  # data.yaml에서 읽기
        "category": "PPE"
    },
    "Hard hat.v1i.yolo26": {
        "classes": ["0 hard hat", "1 no"],
        "category": "PPE"
    },
    "PPE.v14-allinone.yolo26": {
        "classes": None,
        "category": "PPE"
    },
    # Fire 데이터셋
    "fire.v2i.yolo26": {
        "classes": ["fire", "smoke"],
        "category": "Fire"
    },
    "Fire.v2i.yolo26 (1)": {
        "classes": ["fire", "smoke"],
        "category": "Fire"
    },
    "Fire-Final.v1-mitesh.yolo26": {
        "classes": ["PDPU", "fire", "smoke"],
        "category": "Fire"
    },
    # Fall 데이터셋
    "Fall.v1i.yolo26": {
        "classes": ["10-", "down", "person"],
        "category": "Fall"
    },
    "fall.v2i.yolo26": {
        "classes": ["fall", "sit", "walk"],
        "category": "Fall"
    },
    "fall.v4i.yolo26": {
        "classes": ["falling", "sitting", "sleeping", "standing", "walking", "waving hands"],
        "category": "Fall"
    },
}

# 클래스별 색상
COLORS = [
    (0, 255, 0),    # 초록
    (0, 0, 255),    # 빨강
    (255, 0, 0),    # 파랑
    (0, 255, 255),  # 노랑
    (255, 0, 255),  # 마젠타
    (255, 255, 0),  # 청록
    (128, 0, 128),  # 보라
    (0, 128, 255),  # 주황
    (128, 128, 0),  # 올리브
    (0, 128, 128),  # 틸
]


def read_classes_from_yaml(ds_path):
    """data.yaml에서 클래스 목록 읽기"""
    yaml_path = ds_path / "data.yaml"
    if not yaml_path.exists():
        return None

    classes = []
    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # names: ['class1', 'class2'] 형식 파싱
        if "names:" in content:
            import re
            match = re.search(r"names:\s*\[([^\]]+)\]", content)
            if match:
                names_str = match.group(1)
                classes = [n.strip().strip("'\"") for n in names_str.split(",")]
    return classes if classes else None


def draw_boxes(img_path, label_path, class_names):
    """이미지에 bbox 그리기"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, {}

    h, w = img.shape[:2]
    class_counts = defaultdict(int)

    if not label_path.exists():
        return img, class_counts

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                # YOLO format → pixel coordinates
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)

                color = COLORS[cls_id % len(COLORS)]

                if class_names and cls_id < len(class_names):
                    label = f"{cls_id}:{class_names[cls_id]}"
                else:
                    label = f"cls_{cls_id}"

                class_counts[cls_id] += 1

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # 라벨 배경
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1-th-4), (x1+tw, y1), color, -1)
                cv2.putText(img, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            except (ValueError, IndexError):
                continue

    return img, class_counts


def process_dataset(ds_name, ds_info, num_samples=50):
    """데이터셋 처리"""
    ds_path = BASE_DIR / ds_name

    if not ds_path.exists():
        print(f"  [SKIP] 경로 없음")
        return None

    # 클래스 목록 가져오기
    class_names = ds_info.get("classes")
    if class_names is None:
        class_names = read_classes_from_yaml(ds_path)

    # 출력 디렉토리
    safe_name = ds_name.replace(" ", "_").replace("'", "").replace(",", "")[:30]
    out_dir = OUTPUT_DIR / f"{ds_info['category']}_{safe_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_stats = defaultdict(int)
    processed = 0

    for split in ["train", "valid", "test"]:
        img_dir = ds_path / split / "images"
        lbl_dir = ds_path / split / "labels"

        if not img_dir.exists():
            continue

        images = list(img_dir.glob("*.[jJ][pP][gG]")) + list(img_dir.glob("*.[pP][nN][gG]"))

        if not images:
            continue

        # 샘플링
        samples_per_split = num_samples // 3 + 1
        samples = random.sample(images, min(samples_per_split, len(images)))

        for img_path in samples:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            result, counts = draw_boxes(img_path, lbl_path, class_names)

            if result is not None:
                out_path = out_dir / f"{split}_{img_path.name}"
                cv2.imwrite(str(out_path), result)
                processed += 1

                for cls_id, cnt in counts.items():
                    total_stats[cls_id] += cnt

    # 통계 파일 저장
    stats_path = out_dir / "_STATS.txt"
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {ds_name}\n")
        f.write(f"Category: {ds_info['category']}\n")
        f.write(f"Samples: {processed}\n")
        f.write(f"\nClasses:\n")
        if class_names:
            for i, name in enumerate(class_names):
                f.write(f"  [{i}] {name}: {total_stats.get(i, 0)} objects\n")
        else:
            for cls_id, cnt in sorted(total_stats.items()):
                f.write(f"  [{cls_id}] unknown: {cnt} objects\n")

    return {
        "processed": processed,
        "classes": class_names,
        "stats": dict(total_stats),
        "output": out_dir
    }


def main():
    print("="*70)
    print("각 데이터셋별 라벨 품질 검증")
    print("="*70)

    OUTPUT_DIR.mkdir(exist_ok=True)

    results = {}

    for ds_name, ds_info in DATASETS.items():
        print(f"\n[{ds_info['category']}] {ds_name[:50]}...")
        result = process_dataset(ds_name, ds_info, num_samples=50)

        if result:
            print(f"  → {result['processed']}장 저장: {result['output'].name}")
            results[ds_name] = result
        else:
            results[ds_name] = None

    # 요약 출력
    print("\n" + "="*70)
    print("검증 완료!")
    print("="*70)
    print(f"\n결과 위치: {OUTPUT_DIR}")
    print("\n각 폴더를 열어서 바운딩박스가 정확한지 확인하세요:")

    for category in ["PPE", "Fire", "Fall"]:
        print(f"\n[{category}]")
        for ds_name, result in results.items():
            if result and DATASETS[ds_name]["category"] == category:
                print(f"  - {result['output'].name}/")

    print("\n" + "="*70)
    print("확인 사항:")
    print("1. bbox가 객체를 정확히 감싸는지")
    print("2. 클래스 라벨이 올바른지 (숫자:클래스명 형태)")
    print("3. 누락된 객체가 없는지")
    print("4. 각 폴더의 _STATS.txt에서 클래스 분포 확인")
    print("="*70)


if __name__ == "__main__":
    random.seed(42)
    main()
