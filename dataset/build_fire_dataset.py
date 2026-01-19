"""
Fire & Smoke 통합 데이터셋 생성 스크립트
- 2개 클래스: fire(0), smoke(1)
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# 설정
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "fire_smoke_unified"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 클래스 매핑 정의
CLASS_MAPPINGS = {
    "fire.v2i.yolo26": {
        # 원본: fire(0), smoke(1)
        0: 0,   # fire → fire
        1: 1,   # smoke → smoke
    },
    "Fire.v2i.yolo26 (1)": {
        # 원본: fire(0), smoke(1)
        0: 0,   # fire → fire
        1: 1,   # smoke → smoke
    },
    "Fire-Final.v1-mitesh.yolo26": {
        # 원본: PDPU(0), fire(1), smoke(2)
        1: 0,   # fire → fire
        2: 1,   # smoke → smoke
        # PDPU(0)는 제외
    },
    # fire.v2i.yolo26 (2)는 fire/smoke 데이터가 거의 없어 제외
}

# 샘플링 비율 (대용량 데이터셋)
SAMPLE_RATIOS = {
    "Fire-Final.v1-mitesh.yolo26": 0.5,  # 28,937장 → 약 14,000장
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
        print(f"  [SKIP] {ds_name} - 경로 없음")
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
            # 라벨 파일 찾기
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            # 라벨 변환
            new_labels = convert_label(lbl_path, mapping)
            if not new_labels:
                continue  # 유효한 라벨이 없으면 스킵

            # 파일명 prefix (폴더명에서 특수문자 제거)
            prefix = ds_name.replace(" ", "_").replace("(", "").replace(")", "")[:15]
            new_img_name = f"{prefix}_{img_path.name}"
            new_lbl_name = f"{prefix}_{img_path.stem}.txt"

            out_img_path = OUTPUT_DIR / split / "images" / new_img_name
            out_lbl_path = OUTPUT_DIR / split / "labels" / new_lbl_name

            try:
                shutil.copy2(img_path, out_img_path)
                with open(out_lbl_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(new_labels))

                stats[split] += 1

                # 클래스별 통계
                for line in new_labels:
                    cls = int(line.split()[0])
                    stats[f"class_{cls}"] += 1
            except Exception as e:
                print(f"  [ERROR] {img_path}: {e}")

    return stats


def create_data_yaml():
    """data.yaml 생성"""
    yaml_content = """# Fire & Smoke Unified Dataset
# Created for YOLOE fine-tuning

train: ./train/images
val: ./valid/images
test: ./test/images

nc: 2
names: ['fire', 'smoke']

# Class description:
# 0: fire - 화재/불꽃
# 1: smoke - 연기
"""
    with open(OUTPUT_DIR / "data.yaml", 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"data.yaml 생성 완료")


def main():
    print("="*60)
    print("Fire & Smoke 통합 데이터셋 생성")
    print("="*60)

    # 출력 디렉토리 생성
    create_output_dirs()

    total_stats = defaultdict(int)

    # 각 데이터셋 처리
    for ds_name, mapping in CLASS_MAPPINGS.items():
        print(f"\n처리 중: {ds_name}...")

        # 샘플링 비율 적용
        sample_ratio = SAMPLE_RATIOS.get(ds_name, 1.0)

        stats = process_dataset(ds_name, mapping, sample_ratio)

        for k, v in stats.items():
            total_stats[k] += v

        print(f"  완료: train={stats['train']}, valid={stats['valid']}, test={stats['test']}")

    # data.yaml 생성
    create_data_yaml()

    # 최종 통계
    print("\n" + "="*60)
    print("최종 통계")
    print("="*60)
    print(f"Train: {total_stats['train']} images")
    print(f"Valid: {total_stats['valid']} images")
    print(f"Test: {total_stats['test']} images")
    print(f"Total: {total_stats['train'] + total_stats['valid'] + total_stats['test']} images")
    print()
    print("클래스별 객체 수:")
    print(f"  [0] fire: {total_stats['class_0']}")
    print(f"  [1] smoke: {total_stats['class_1']}")
    print()
    print(f"데이터셋 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
