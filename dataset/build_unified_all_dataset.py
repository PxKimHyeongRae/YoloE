"""
PPE + Fire + Fall 전체 통합 데이터셋 생성 스크립트
- 8개 클래스:
  0: Helmet_OFF, 1: Helmet_ON, 2: Vest_OFF, 3: Vest_ON
  4: fire, 5: smoke
  6: fall (down, fall, falling, sleeping)
  7: person (person, standing, walking, sitting)
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# 설정
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "unified_all"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ============================================================
# PPE 데이터셋 클래스 매핑 (0-3)
# ============================================================
PPE_CLASS_MAPPINGS = {
    "construction safety.v2-release.yolo26_helmet_no-helmet_no-vest_vest_person": {
        0: 1,   # helmet → Helmet_ON
        1: 0,   # no-helmet → Helmet_OFF
        2: 2,   # no-vest → Vest_OFF
        4: 3,   # vest → Vest_ON
    },
    "ppe_only_700_Hard_Hat_OFF', 'Hard_Hat_ON', 'Safety_Vest_OFF', 'Safety_Vest_ON'": {
        0: 0,   # Hard_Hat_OFF → Helmet_OFF
        1: 1,   # Hard_Hat_ON → Helmet_ON
        2: 2,   # Safety_Vest_OFF → Vest_OFF
        3: 3,   # Safety_Vest_ON → Vest_ON
    },
    "ppe_vest_13k": {
        0: 2,   # no_safety_vest → Vest_OFF
        1: 3,   # safety_vest → Vest_ON
    },
    "Safety vest.v3i.yolo26_vest_no-vest_6k": {
        0: 2,   # NO-Safety Vest → Vest_OFF
        1: 3,   # Safety Vest → Vest_ON
    },
    "PPE-0.3.v1i.yolo26": {
        3: 1,   # Helmet → Helmet_ON
        12: 0,  # Without Helmet → Helmet_OFF
        8: 3,   # Vest → Vest_ON
        15: 2,  # Without Vest → Vest_OFF
    },
    "Hard hat.v1i.yolo26": {
        0: 1,   # 0 hard hat → Helmet_ON
        1: 0,   # 1 no → Helmet_OFF
    },
    "PPE.v14-allinone.yolo26": {
        2: 1,   # hardhat → Helmet_ON
        5: 0,   # no_hardhat → Helmet_OFF
        8: 3,   # vest → Vest_ON
        6: 2,   # no_vest → Vest_OFF
    },
}

# ============================================================
# Fire 데이터셋 클래스 매핑 (4-5)
# ============================================================
FIRE_CLASS_MAPPINGS = {
    "fire.v2i.yolo26": {
        0: 4,   # fire → fire
        1: 5,   # smoke → smoke
    },
    "Fire.v2i.yolo26 (1)": {
        0: 4,   # fire → fire
        1: 5,   # smoke → smoke
    },
    "Fire-Final.v1-mitesh.yolo26": {
        1: 4,   # fire → fire
        2: 5,   # smoke → smoke
    },
}

# ============================================================
# Fall 데이터셋 클래스 매핑 (6: fall, 7: person)
# ============================================================
FALL_CLASS_MAPPINGS = {
    "Fall.v1i.yolo26": {
        # 원본: 10-(0), down(1), person(2)
        1: 6,   # down → fall
        2: 7,   # person → person
        # 0: 10- 제외 (의미 불명)
    },
    "fall.v2i.yolo26": {
        # 원본: fall(0), sit(1), walk(2)
        0: 6,   # fall → fall
        1: 7,   # sit → person
        2: 7,   # walk → person
    },
    "fall.v4i.yolo26": {
        # 원본: falling(0), sitting(1), sleeping(2), standing(3), walking(4), waving hands(5)
        0: 6,   # falling → fall
        1: 7,   # sitting → person
        2: 6,   # sleeping → fall (누워있는 상태는 위험)
        3: 7,   # standing → person
        4: 7,   # walking → person
        # 5: waving hands 제외 (소량, 관련 없음)
    },
}

# ============================================================
# 샘플링 비율
# ============================================================
SAMPLE_RATIOS = {
    # PPE
    "ppe_vest_13k": 0.3,
    "Safety vest.v3i.yolo26_vest_no-vest_6k": 0.3,
    "Hard hat.v1i.yolo26": 0.3,
    "PPE-0.3.v1i.yolo26": 0.5,
    # Fire
    "Fire-Final.v1-mitesh.yolo26": 0.5,
}


def create_output_dirs():
    """출력 디렉토리 생성"""
    for split in ["train", "valid", "test"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    print(f"출력 디렉토리 생성: {OUTPUT_DIR}")


def convert_label(label_path, mapping):
    """라벨 파일 변환 (클래스 매핑 적용)"""
    new_lines = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            orig_cls = int(parts[0])
            if orig_cls in mapping:
                new_cls = mapping[orig_cls]
                new_line = f"{new_cls} " + " ".join(parts[1:])
                new_lines.append(new_line)
    return new_lines


def process_dataset(ds_name, mapping, sample_ratio=1.0):
    """데이터셋 처리"""
    ds_path = BASE_DIR / ds_name
    if not ds_path.exists():
        print(f"  [SKIP] {ds_name[:40]}... - 경로 없음")
        return defaultdict(int)

    stats = defaultdict(int)

    for split in ["train", "valid", "test"]:
        img_dir = ds_path / split / "images"
        lbl_dir = ds_path / split / "labels"

        if not img_dir.exists():
            continue

        # 이미지 파일 목록
        images = list(img_dir.glob("*.[jJ][pP][gG]")) + list(img_dir.glob("*.[pP][nN][gG]"))

        # 샘플링
        if sample_ratio < 1.0:
            images = random.sample(images, int(len(images) * sample_ratio))

        for img_path in images:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            new_labels = convert_label(lbl_path, mapping)
            if not new_labels:
                continue

            # 파일명 prefix
            prefix = ds_name.replace(" ", "_").replace("(", "").replace(")", "").replace("'", "")[:15]
            new_img_name = f"{prefix}_{img_path.name}"
            new_lbl_name = f"{prefix}_{img_path.stem}.txt"

            out_img_path = OUTPUT_DIR / split / "images" / new_img_name
            out_lbl_path = OUTPUT_DIR / split / "labels" / new_lbl_name

            try:
                shutil.copy2(img_path, out_img_path)
                with open(out_lbl_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(new_labels))

                stats[split] += 1

                for line in new_labels:
                    cls = int(line.split()[0])
                    stats[f"class_{cls}"] += 1
            except Exception as e:
                pass  # 에러 무시

    return stats


def create_data_yaml():
    """data.yaml 생성"""
    yaml_content = """# PPE + Fire + Fall Unified Dataset
# Created for YOLOE fine-tuning

train: ./train/images
val: ./valid/images
test: ./test/images

nc: 8
names: ['Helmet_OFF', 'Helmet_ON', 'Vest_OFF', 'Vest_ON', 'fire', 'smoke', 'fall', 'person']

# Class description:
# 0: Helmet_OFF - 헬멧 미착용
# 1: Helmet_ON - 헬멧 착용
# 2: Vest_OFF - 안전조끼 미착용
# 3: Vest_ON - 안전조끼 착용
# 4: fire - 화재/불꽃
# 5: smoke - 연기
# 6: fall - 쓰러짐/넘어짐 (down, fall, falling, sleeping)
# 7: person - 정상 자세 (person, standing, walking, sitting)
"""
    with open(OUTPUT_DIR / "data.yaml", 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"data.yaml 생성 완료")


def main():
    print("="*60)
    print("PPE + Fire 전체 통합 데이터셋 생성")
    print("="*60)

    create_output_dirs()

    total_stats = defaultdict(int)

    # PPE 데이터셋 처리
    print("\n[PPE 데이터셋 처리]")
    for ds_name, mapping in PPE_CLASS_MAPPINGS.items():
        print(f"  처리 중: {ds_name[:45]}...")
        sample_ratio = SAMPLE_RATIOS.get(ds_name, 1.0)
        stats = process_dataset(ds_name, mapping, sample_ratio)
        for k, v in stats.items():
            total_stats[k] += v
        print(f"    → train={stats['train']}, valid={stats['valid']}, test={stats['test']}")

    # Fire 데이터셋 처리
    print("\n[Fire 데이터셋 처리]")
    for ds_name, mapping in FIRE_CLASS_MAPPINGS.items():
        print(f"  처리 중: {ds_name}...")
        sample_ratio = SAMPLE_RATIOS.get(ds_name, 1.0)
        stats = process_dataset(ds_name, mapping, sample_ratio)
        for k, v in stats.items():
            total_stats[k] += v
        print(f"    → train={stats['train']}, valid={stats['valid']}, test={stats['test']}")

    # Fall 데이터셋 처리
    print("\n[Fall 데이터셋 처리]")
    for ds_name, mapping in FALL_CLASS_MAPPINGS.items():
        print(f"  처리 중: {ds_name}...")
        sample_ratio = SAMPLE_RATIOS.get(ds_name, 1.0)
        stats = process_dataset(ds_name, mapping, sample_ratio)
        for k, v in stats.items():
            total_stats[k] += v
        print(f"    → train={stats['train']}, valid={stats['valid']}, test={stats['test']}")

    create_data_yaml()

    # 최종 통계
    print("\n" + "="*60)
    print("최종 통계")
    print("="*60)
    total = total_stats['train'] + total_stats['valid'] + total_stats['test']
    print(f"Train: {total_stats['train']} images")
    print(f"Valid: {total_stats['valid']} images")
    print(f"Test: {total_stats['test']} images")
    print(f"Total: {total} images")
    print()
    print("클래스별 객체 수:")
    class_names = ['Helmet_OFF', 'Helmet_ON', 'Vest_OFF', 'Vest_ON', 'fire', 'smoke', 'fall', 'person']
    for i, name in enumerate(class_names):
        print(f"  [{i}] {name}: {total_stats[f'class_{i}']}")
    print()
    print(f"데이터셋 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
