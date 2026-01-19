"""
YOLOE Fire & Smoke 파인튜닝 스크립트

사용법:
    python train_fire_yoloe.py                    # 기본 설정으로 학습
    python train_fire_yoloe.py --model yoloe-26s-seg.pt  # 다른 모델 사용
    python train_fire_yoloe.py --epochs 200       # 에폭 변경
    python train_fire_yoloe.py --resume           # 이어서 학습

클래스:
    0: fire (화재/불꽃)
    1: smoke (연기)
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Fire & Smoke YOLO Training")

    # 모델 설정
    parser.add_argument("--model", type=str, default="yoloe-26m-seg.pt",
                        help="모델 파일 (yoloe-26n/s/m/l/x-seg.pt)")
    parser.add_argument("--data", type=str, default="dataset/fire_smoke_unified/data.yaml",
                        help="데이터셋 yaml 경로")

    # 학습 설정
    parser.add_argument("--epochs", type=int, default=100, help="학습 에폭 수")
    parser.add_argument("--batch", type=int, default=16, help="배치 크기")
    parser.add_argument("--imgsz", type=int, default=640, help="이미지 크기")
    parser.add_argument("--device", type=str, default="", help="GPU 디바이스 (0, 0,1, cpu)")

    # 프로젝트 설정
    parser.add_argument("--project", type=str, default="runs/fire_smoke", help="프로젝트 폴더")
    parser.add_argument("--name", type=str, default="", help="실험 이름 (자동 생성)")

    # 기타
    parser.add_argument("--resume", action="store_true", help="마지막 학습에서 이어서 학습")
    parser.add_argument("--val-only", action="store_true", help="검증만 실행")
    parser.add_argument("--export", type=str, default="", help="모델 내보내기 (onnx, torchscript, etc)")

    return parser.parse_args()


def train(args):
    """모델 학습"""
    print("="*60)
    print("Fire & Smoke Detection Training")
    print("="*60)

    # 모델 로드
    if args.resume:
        last_pt = Path(args.project) / "*/weights/last.pt"
        import glob
        last_files = glob.glob(str(last_pt))
        if last_files:
            model_path = max(last_files, key=lambda x: Path(x).stat().st_mtime)
            print(f"이어서 학습: {model_path}")
            model = YOLO(model_path)
        else:
            print("이전 학습 파일 없음. 새로 시작합니다.")
            model = YOLO(args.model)
    else:
        print(f"모델 로드: {args.model}")
        model = YOLO(args.model)

    # 실험 이름 생성
    if not args.name:
        model_name = Path(args.model).stem
        args.name = f"{model_name}_e{args.epochs}_b{args.batch}"

    print(f"데이터셋: {args.data}")
    print(f"설정: epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}")
    print(f"저장 위치: {args.project}/{args.name}")
    print("="*60)

    # 학습 실행
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device if args.device else None,

        # 최적화 설정
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,

        # 데이터 증강
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,

        # 저장 설정
        patience=30,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "="*60)
    print("학습 완료!")
    print("="*60)
    print(f"Best model: {args.project}/{args.name}/weights/best.pt")
    print(f"Last model: {args.project}/{args.name}/weights/last.pt")

    return model


def validate(args):
    """모델 검증"""
    print("검증 모드")

    best_pt = Path(args.project) / "*/weights/best.pt"
    import glob
    best_files = glob.glob(str(best_pt))

    if not best_files:
        print("학습된 모델이 없습니다. 먼저 학습을 실행하세요.")
        return

    model_path = max(best_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"모델 로드: {model_path}")

    model = YOLO(model_path)
    metrics = model.val(data=args.data)

    print("\n" + "="*60)
    print("검증 결과")
    print("="*60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print("\n클래스별 AP50:")
    class_names = ['fire', 'smoke']
    for i, ap in enumerate(metrics.box.ap50):
        print(f"  {class_names[i]}: {ap:.4f}")


def export_model(args):
    """모델 내보내기"""
    print(f"모델 내보내기: {args.export}")

    best_pt = Path(args.project) / "*/weights/best.pt"
    import glob
    best_files = glob.glob(str(best_pt))

    if not best_files:
        print("학습된 모델이 없습니다.")
        return

    model_path = max(best_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"모델 로드: {model_path}")

    model = YOLO(model_path)
    model.export(format=args.export)
    print("내보내기 완료!")


def main():
    args = parse_args()

    if args.val_only:
        validate(args)
    elif args.export:
        export_model(args)
    else:
        model = train(args)

        print("\n학습 후 검증 실행...")
        metrics = model.val()
        print(f"\n최종 mAP50: {metrics.box.map50:.4f}")
        print(f"최종 mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
