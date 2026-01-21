"""
YOLOE 통합 모델 파인튜닝 스크립트 (Helmet + Fire + Fall)

사용법:
    python train_unified.py                       # 기본 설정으로 학습
    python train_unified.py --model yoloe-26l     # 다른 모델 사용
    python train_unified.py --epochs 150          # 에폭 변경

클래스 (6개):
    0: Helmet_OFF (헬멧 미착용)
    1: Helmet_ON (헬멧 착용)
    2: fire (화재/불꽃)
    3: smoke (연기)
    4: fall (쓰러짐/넘어짐)
    5: person (사람)
"""

import argparse
from pathlib import Path
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer


def parse_args():
    parser = argparse.ArgumentParser(description="PPE + Fire + Fall Unified YOLOE Training")

    # 모델 설정 (n, s, m, l, x)
    parser.add_argument("--model", type=str, default="yoloe-26l",
                        help="모델 크기 (yoloe-26n, yoloe-26s, yoloe-26m, yoloe-26l, yoloe-26x)")
    parser.add_argument("--data", type=str, default="dataset/unified_no_vest/data.yaml",
                        help="데이터셋 yaml 경로")

    # 학습 설정
    parser.add_argument("--epochs", type=int, default=100, help="학습 에폭 수")
    parser.add_argument("--batch", type=int, default=32, help="배치 크기")
    parser.add_argument("--imgsz", type=int, default=640, help="이미지 크기")
    parser.add_argument("--device", type=int, default=0, help="GPU 디바이스")

    # 프로젝트 설정
    parser.add_argument("--project", type=str, default="runs/unified", help="프로젝트 폴더")
    parser.add_argument("--name", type=str, default="", help="실험 이름 (자동 생성)")

    # 기타
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")

    return parser.parse_args()


def train(args):
    """모델 학습"""
    print("="*60)
    print("Helmet + Fire + Fall Unified Detection Training (YOLOE)")
    print("="*60)
    print("클래스: Helmet_OFF, Helmet_ON, fire, smoke, fall, person")
    print("="*60)

    BASE_PATH = Path(__file__).parent

    # 모델 설정
    model_name = args.model  # e.g., "yoloe-26m"
    model_yaml = f"{model_name}.yaml"
    model_weights = BASE_PATH / "models" / f"{model_name}-seg.pt"

    # 데이터 경로
    data_path = BASE_PATH / args.data

    # 실험 이름 생성
    if not args.name:
        args.name = f"{model_name}_e{args.epochs}_b{args.batch}"

    print(f"Model YAML: {model_yaml}")
    print(f"Weights: {model_weights}")
    print(f"Data: {data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Output: {args.project}/{args.name}")
    print("="*60)

    # 가중치 파일 확인
    if not model_weights.exists():
        print(f"[ERROR] 가중치 파일이 없습니다: {model_weights}")
        print("사용 가능한 모델:")
        for pt in (BASE_PATH / "models").glob("yoloe-*.pt"):
            print(f"  - {pt.stem.replace('-seg', '')}")
        return None

    # YOLOE 모델 초기화 (YAML 설정 사용)
    model = YOLOE(model_yaml)

    # 사전 학습 가중치 로드
    model.load(str(model_weights))

    # 학습 설정
    train_config = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "name": args.name,
        "project": str(BASE_PATH / args.project),
        "patience": args.patience,
        "device": args.device,
        "trainer": YOLOEPETrainer,
        # 최적화 설정
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        # 데이터 증강
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
        # 저장 설정
        "save": True,
        "save_period": 10,
        "val": True,
        "plots": True,
        "verbose": True,
    }

    # 학습 실행
    results = model.train(**train_config)

    print("\n" + "="*60)
    print("학습 완료!")
    print("="*60)
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print(f"Last model: {results.save_dir}/weights/last.pt")

    return model, results


def main():
    args = parse_args()
    result = train(args)

    if result:
        model, results = result
        print(f"\n최종 결과 저장 위치: {results.save_dir}")


if __name__ == "__main__":
    main()
