"""
Fall/Fire 데이터셋 상세 분석
- 각 클래스별 객체 수
- 분포 확인
- 학습 가능성 판단
"""

from pathlib import Path
from collections import defaultdict
import os

BASE_DIR = Path(__file__).parent / "dataset"

# 분석할 데이터셋
DATASETS = {
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
}


def analyze_dataset(ds_name, ds_info):
    """데이터셋 분석"""
    ds_path = BASE_DIR / ds_name

    if not ds_path.exists():
        return None

    class_names = ds_info["classes"]

    stats = {
        "total_images": 0,
        "total_objects": 0,
        "class_counts": defaultdict(int),
        "splits": {}
    }

    for split in ["train", "valid", "test"]:
        img_dir = ds_path / split / "images"
        lbl_dir = ds_path / split / "labels"

        if not img_dir.exists():
            continue

        images = list(img_dir.glob("*.[jJ][pP][gG]")) + list(img_dir.glob("*.[pP][nN][gG]"))
        split_counts = defaultdict(int)

        for img_path in images:
            lbl_path = lbl_dir / (img_path.stem + ".txt")

            if lbl_path.exists():
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(parts[0])
                                stats["class_counts"][cls_id] += 1
                                split_counts[cls_id] += 1
                                stats["total_objects"] += 1
                            except ValueError:
                                pass

        stats["total_images"] += len(images)
        stats["splits"][split] = {
            "images": len(images),
            "counts": dict(split_counts)
        }

    return stats


def print_analysis():
    """분석 결과 출력"""

    print("=" * 80)
    print("FALL / FIRE 데이터셋 상세 분석")
    print("=" * 80)

    for category in ["Fall", "Fire"]:
        print(f"\n{'─' * 80}")
        print(f"[{category.upper()} 데이터셋]")
        print(f"{'─' * 80}")

        for ds_name, ds_info in DATASETS.items():
            if ds_info["category"] != category:
                continue

            print(f"\n[Dataset] {ds_name}")

            stats = analyze_dataset(ds_name, ds_info)

            if stats is None:
                print("   ❌ 경로 없음")
                continue

            class_names = ds_info["classes"]

            print(f"   총 이미지: {stats['total_images']:,}장")
            print(f"   총 객체: {stats['total_objects']:,}개")

            print(f"\n   클래스별 분포:")
            print(f"   {'클래스':<20} {'개수':>10} {'비율':>10} {'바'}")
            print(f"   {'-'*50}")

            for i, name in enumerate(class_names):
                count = stats["class_counts"].get(i, 0)
                pct = count / max(1, stats["total_objects"]) * 100
                bar = "#" * int(pct / 2)

                # 현재 사용 여부 표시
                if category == "Fall":
                    if name in ["down", "fall", "falling"]:
                        marker = "✅ 사용중"
                    else:
                        marker = "❓ 미사용"
                else:  # Fire
                    if name in ["fire", "smoke"]:
                        marker = "✅ 사용중"
                    else:
                        marker = "❓ 미사용"

                print(f"   [{i}] {name:<15} {count:>10,} {pct:>8.1f}%  {bar} {marker}")

            # Split별 분포
            print(f"\n   Split별 이미지:")
            for split, split_data in stats["splits"].items():
                print(f"     {split}: {split_data['images']:,}장")

    # 요약 및 권장사항
    print("\n" + "=" * 80)
    print("분석 요약 및 권장사항")
    print("=" * 80)

    print("\n[FALL 데이터셋 분석]")
    print("""
    현재 사용: down, fall, falling → 통합하여 'fall' 클래스

    미사용 클래스 검토:
    - person: 서있는 사람 (정상 상태)
    - sit: 앉은 사람 (정상 상태)
    - walk: 걷는 사람 (정상 상태)
    - standing: 서있는 사람 (정상 상태)
    - walking: 걷는 사람 (정상 상태)
    - sitting: 앉은 사람 (정상 상태)
    - sleeping: 누워있는 사람 (fall과 유사?)
    - waving hands: 손 흔드는 사람
    - 10-: 의미 불명

    질문:
    1. sleeping은 fall에 포함해야 하나? (누워있음)
    2. standing/walking/sitting을 정상 상태로 학습해야 하나?
    """)

    print("\n[FIRE 데이터셋 분석]")
    print("""
    현재 사용: fire, smoke
    미사용: PDPU (의미 불명 - 제외해도 될 듯)

    → Fire는 현재 방식 유지 권장
    """)


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    print_analysis()
