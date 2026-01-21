"""
YOLOE 통합 모델 파인튜닝 스크립트 (Helmet + Fire + Fall)

사용법:
    python train_unified.py                              # Linear Probing (기본, 10 epochs)
    python train_unified.py --mode full                  # Full Tuning (80 epochs)
    python train_unified.py --model yoloe-26l            # 다른 모델 사용
    python train_unified.py --mode full --epochs 100     # Full Tuning + 에폭 변경

학습 모드:
    - linear (기본): Linear Probing - 마지막 레이어만 학습 (10 epochs)
                     빠른 실험/검증에 적합, 80%+ 성능 달성 가능
    - full: Full Tuning - 전체 파라미터 학습 (80 epochs)
            최고 성능을 위한 완전한 fine-tuning

클래스 (6개) - CLIP 친화적 자연어 명칭:
    0: person without helmet (헬멧 미착용)
    1: person wearing helmet (헬멧 착용)
    2: fire (화재/불꽃)
    3: smoke (연기)
    4: fallen person (쓰러진 사람)
    5: person (일반 사람)
"""

import argparse
from pathlib import Path
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer


# 클래스 정의 (자연어 기반)
CLASS_NAMES = [
    'person without helmet',    # 0
    'person wearing helmet',    # 1
    'fire',                     # 2
    'smoke',                    # 3
    'fallen person',            # 4
    'person',                   # 5
]


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOE Fine-tuning for Safety Detection")

    # 학습 모드
    parser.add_argument("--mode", type=str, default="linear",
                        choices=["linear", "full"],
                        help="학습 모드: linear (Linear Probing, 10 epochs) / full (Full Tuning, 80 epochs)")

    # 모델 설정
    parser.add_argument("--model", type=str, default="yoloe-26l",
                        help="모델 크기 (yoloe-26s, yoloe-26m, yoloe-26l)")
    parser.add_argument("--data", type=str, default="dataset/unified_no_vest/data.yaml",
                        help="데이터셋 yaml 경로")

    # 학습 설정 (mode에 따라 기본값 결정)
    parser.add_argument("--epochs", type=int, default=None,
                        help="학습 에폭 수 (기본: linear=10, full=80)")
    parser.add_argument("--batch", type=int, default=32, help="배치 크기")
    parser.add_argument("--imgsz", type=int, default=640, help="이미지 크기")
    parser.add_argument("--device", type=int, default=0, help="GPU 디바이스")

    # 프로젝트 설정
    parser.add_argument("--project", type=str, default="runs/unified", help="프로젝트 폴더")
    parser.add_argument("--name", type=str, default="", help="실험 이름 (자동 생성)")

    # 기타
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience (기본: linear=5, full=20)")

    # 기존 가중치에서 시작 (선택)
    parser.add_argument("--weights", type=str, default="",
                        help="기존 학습된 가중치 파일 경로 (예: runs/unified/exp/weights/best.pt)")

    return parser.parse_args()


def train(args):
    """모델 학습"""
    print("=" * 60)
    print("YOLOE Safety Detection Training")
    print("=" * 60)

    # 모드별 기본값 설정
    if args.mode == "linear":
        default_epochs = 10
        default_patience = 5
        mode_desc = "Linear Probing (마지막 레이어만 학습)"
    else:  # full
        default_epochs = 80
        default_patience = 20
        mode_desc = "Full Tuning (전체 파라미터 학습)"

    # 사용자 지정값 또는 기본값 사용
    epochs = args.epochs if args.epochs is not None else default_epochs
    patience = args.patience if args.patience is not None else default_patience

    print(f"Mode: {args.mode.upper()} - {mode_desc}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print("=" * 60)

    BASE_PATH = Path(__file__).parent

    # 모델 설정
    model_name = args.model
    model_yaml = f"{model_name}.yaml"
    model_weights = BASE_PATH / "models" / f"{model_name}-seg.pt"

    # 데이터 경로
    data_path = BASE_PATH / args.data

    # 실험 이름 생성
    if not args.name:
        args.name = f"{model_name}_{args.mode}_e{epochs}_b{args.batch}"

    print(f"Model: {model_name}")
    print(f"Weights: {model_weights}")
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch: {args.batch}")
    print(f"Patience: {patience}")
    print(f"Output: {args.project}/{args.name}")
    print("=" * 60)

    # 가중치 로드
    if args.weights:
        # 기존 학습된 가중치 사용
        weights_path = Path(args.weights)
        if not weights_path.is_absolute():
            weights_path = BASE_PATH / weights_path
        if not weights_path.exists():
            print(f"[ERROR] 가중치 파일이 없습니다: {weights_path}")
            return None
        print(f"[INFO] 기존 가중치 로드: {weights_path}")
        model = YOLOE(str(weights_path))
    else:
        # 사전 학습 가중치 사용
        if not model_weights.exists():
            print(f"[ERROR] 가중치 파일이 없습니다: {model_weights}")
            print("사용 가능한 모델:")
            models_dir = BASE_PATH / "models"
            if models_dir.exists():
                for pt in models_dir.glob("yoloe-*.pt"):
                    print(f"  - {pt.stem.replace('-seg', '')}")
            return None

        # YOLOE 모델 초기화 (YAML 설정 사용)
        model = YOLOE(model_yaml)
        model.load(str(model_weights))

    # 학습 설정
    train_config = {
        "data": str(data_path),
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "name": args.name,
        "project": str(BASE_PATH / args.project),
        "patience": patience,
        "device": args.device,
        "trainer": YOLOEPETrainer,
        # 저장 설정
        "save": True,
        "save_period": 10 if args.mode == "full" else -1,
        "val": True,
        "plots": True,
        "verbose": True,
    }

    # 모드별 최적화 설정
    if args.mode == "linear":
        # Linear Probing: 높은 학습률, 짧은 warmup
        train_config.update({
            "optimizer": "AdamW",
            "lr0": 0.002,       # 높은 학습률 (마지막 레이어만)
            "lrf": 0.1,
            "weight_decay": 0.025,
            "warmup_epochs": 1,
            # 약한 augmentation
            "hsv_h": 0.01,
            "hsv_s": 0.5,
            "hsv_v": 0.3,
            "degrees": 5.0,
            "translate": 0.1,
            "scale": 0.3,
            "shear": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.5,
            "mixup": 0.0,
            # Linear Probing 특화 설정
            "freeze": None,  # YOLOEPETrainer가 자동으로 freeze 처리
        })
    else:  # full
        # Full Tuning: 낮은 학습률, 강한 augmentation
        train_config.update({
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            # 강한 augmentation
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 10.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 2.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.1,
        })

    # 학습 실행
    results = model.train(**train_config)

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print(f"Last model: {results.save_dir}/weights/last.pt")

    # 다음 단계 안내
    if args.mode == "linear":
        print("\n[다음 단계]")
        print("Linear Probing 결과가 만족스러우면 Full Tuning 진행:")
        print(f"  python train_unified.py --mode full --weights {results.save_dir}/weights/best.pt")

    return model, results


def main():
    args = parse_args()
    result = train(args)

    if result:
        model, results = result
        print(f"\n최종 결과 저장 위치: {results.save_dir}")


if __name__ == "__main__":
    main()
