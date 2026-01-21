"""
YOLOE 추론 스크립트 (이미지/폴더/RTSP 지원)

사용법:
    python inference.py --source image.jpg                    # 단일 이미지
    python inference.py --source ./images/                    # 폴더 내 모든 이미지
    python inference.py --source rtsp://192.168.1.100:554/stream  # RTSP 스트림
    python inference.py --source 0                            # 웹캠 (디바이스 0)

옵션:
    --model      : 모델 가중치 경로 (기본: runs/unified/yoloe-26m_e100_b16/weights/best.pt)
    --conf       : 신뢰도 임계값 (기본: 0.25)
    --iou        : NMS IOU 임계값 (기본: 0.45)
    --imgsz      : 입력 이미지 크기 (기본: 640)
    --save       : 결과 이미지 저장
    --show       : 실시간 화면 표시
    --output     : 결과 저장 폴더 (기본: runs/detect)
"""

import argparse
import cv2
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLOE


# 클래스별 색상 정의 (BGR)
CLASS_COLORS = {
    0: (0, 0, 255),      # Helmet_OFF - 빨강
    1: (0, 255, 0),      # Helmet_ON - 초록
    2: (0, 128, 255),    # Vest_OFF - 주황
    3: (0, 255, 255),    # Vest_ON - 노랑
    4: (0, 0, 200),      # fire - 진한 빨강
    5: (128, 128, 128),  # smoke - 회색
    6: (255, 0, 255),    # fall - 마젠타
    7: (255, 255, 0),    # person - 시안
}

CLASS_NAMES = {
    0: "Helmet_OFF",
    1: "Helmet_ON",
    2: "Vest_OFF",
    3: "Vest_ON",
    4: "fire",
    5: "smoke",
    6: "fall",
    7: "person",
}


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOE Inference Script")

    parser.add_argument("--source", type=str, required=True,
                        help="입력 소스 (이미지 파일, 폴더 경로, RTSP URL, 또는 웹캠 번호)")
    parser.add_argument("--model", type=str,
                        default="best.pt",
                        help="모델 가중치 경로 (기본: 현재 폴더의 best.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="신뢰도 임계값 (0-1)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IOU 임계값")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="입력 이미지 크기")
    parser.add_argument("--save", action="store_true",
                        help="결과 이미지 저장")
    parser.add_argument("--show", action="store_true",
                        help="실시간 화면 표시")
    parser.add_argument("--output", type=str, default="runs/detect",
                        help="결과 저장 폴더")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU 디바이스 번호")
    parser.add_argument("--line-width", type=int, default=2,
                        help="바운딩박스 선 두께")
    parser.add_argument("--font-scale", type=float, default=0.6,
                        help="폰트 크기")
    parser.add_argument("--sample", type=int, default=0,
                        help="폴더 모드에서 샘플링할 이미지 개수 (0=전체)")
    parser.add_argument("--random", action="store_true",
                        help="샘플링 시 랜덤 선택 (기본: 정렬순)")

    return parser.parse_args()


def draw_detections(image, results, line_width=2, font_scale=0.6):
    """
    검출 결과를 이미지에 그리기
    """
    annotated = image.copy()

    if results[0].boxes is None or len(results[0].boxes) == 0:
        return annotated, []

    boxes = results[0].boxes
    detections = []

    for i, box in enumerate(boxes):
        # 바운딩박스 좌표
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # 클래스와 신뢰도
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        # 클래스 이름과 색상
        cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))

        # 바운딩박스 그리기
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)

        # 라벨 텍스트
        label = f"{cls_name} {conf:.2f}"

        # 텍스트 배경
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - 10),
            (x1 + text_w + 5, y1),
            color,
            -1
        )

        # 텍스트 그리기
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2
        )

        detections.append({
            "class": cls_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    return annotated, detections


def process_image(model, image_path, args, output_dir=None):
    """
    단일 이미지 처리
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] 이미지를 읽을 수 없습니다: {image_path}")
        return None

    # 추론 실행
    results = model.predict(
        image,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False
    )

    # 결과 그리기
    annotated, detections = draw_detections(
        image, results, args.line_width, args.font_scale
    )

    # 결과 출력
    print(f"\n[{Path(image_path).name}] 검출 결과:")
    if detections:
        for det in detections:
            print(f"  - {det['class']}: {det['confidence']:.2%} "
                  f"@ [{det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}]")
    else:
        print("  검출된 객체 없음")

    # 저장
    if args.save and output_dir:
        save_path = output_dir / Path(image_path).name
        cv2.imwrite(str(save_path), annotated)
        print(f"  저장: {save_path}")

    # 화면 표시
    if args.show:
        cv2.imshow("YOLOE Detection", annotated)
        cv2.waitKey(0)

    return annotated, detections


def process_folder(model, folder_path, args, output_dir):
    """
    폴더 내 모든 이미지 처리
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    folder = Path(folder_path)

    image_files = [
        f for f in folder.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"[ERROR] 폴더에 이미지 파일이 없습니다: {folder_path}")
        return

    total_count = len(image_files)

    # 샘플링 처리
    if args.sample > 0 and args.sample < total_count:
        if args.random:
            image_files = random.sample(image_files, args.sample)
            print(f"\n총 {total_count}개 중 {args.sample}개 랜덤 샘플링")
        else:
            image_files = sorted(image_files)[:args.sample]
            print(f"\n총 {total_count}개 중 앞에서 {args.sample}개 선택")
    else:
        image_files = sorted(image_files)
        print(f"\n총 {total_count}개 이미지 처리 중...")

    print("=" * 60)

    total_detections = 0
    process_count = len(image_files)

    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{process_count}] 처리 중: {img_path.name}")
        result = process_image(model, img_path, args, output_dir)
        if result:
            _, detections = result
            total_detections += len(detections)

    print("\n" + "=" * 60)
    print(f"처리 완료: {process_count}개 이미지, 총 {total_detections}개 검출")
    if args.sample > 0:
        print(f"(전체 {total_count}개 중 {process_count}개 샘플링됨)")
    if args.save:
        print(f"결과 저장 위치: {output_dir}")


def process_stream(model, source, args, output_dir):
    """
    비디오 스트림 (RTSP/웹캠) 처리
    """
    # 웹캠 번호인 경우 정수로 변환
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] 스트림을 열 수 없습니다: {source}")
        return

    print(f"\n스트림 연결됨: {source}")
    print("종료하려면 'q' 키를 누르세요")
    print("=" * 60)

    frame_count = 0
    fps_start_time = datetime.now()

    # 비디오 저장 설정
    video_writer = None
    if args.save:
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = output_dir / f"stream_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        print(f"비디오 저장: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n프레임을 읽을 수 없습니다. 재연결 시도...")
                cap.release()
                cap = cv2.VideoCapture(source)
                continue

            frame_count += 1

            # 추론 실행
            results = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False
            )

            # 결과 그리기
            annotated, detections = draw_detections(
                frame, results, args.line_width, args.font_scale
            )

            # FPS 계산
            elapsed = (datetime.now() - fps_start_time).total_seconds()
            if elapsed > 0:
                fps = frame_count / elapsed
            else:
                fps = 0

            # FPS 표시
            cv2.putText(
                annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # 검출 수 표시
            cv2.putText(
                annotated, f"Detections: {len(detections)}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # 비디오 저장
            if video_writer:
                video_writer.write(annotated)

            # 화면 표시
            if args.show:
                cv2.imshow("YOLOE Detection (Press 'q' to quit)", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 's' 키로 현재 프레임 저장
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = output_dir / f"capture_{timestamp}.jpg"
                    cv2.imwrite(str(save_path), annotated)
                    print(f"\n캡처 저장: {save_path}")

            # 주기적 상태 출력
            if frame_count % 100 == 0:
                print(f"Frame {frame_count}: {len(detections)} 검출, {fps:.1f} FPS")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단됨")

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print(f"\n총 {frame_count} 프레임 처리됨")


def main():
    args = parse_args()

    # 출력 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / timestamp
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    print("=" * 60)
    print("YOLOE Inference")
    print("=" * 60)
    print(f"모델: {args.model}")
    print(f"신뢰도 임계값: {args.conf}")
    print(f"IOU 임계값: {args.iou}")
    print(f"이미지 크기: {args.imgsz}")
    print("=" * 60)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 모델 파일이 없습니다: {args.model}")
        return

    print("\n모델 로딩 중...")
    model = YOLOE(str(model_path))
    print("모델 로드 완료!")

    source = args.source
    source_path = Path(source) if not source.startswith(('rtsp://', 'http://', 'https://')) and not source.isdigit() else None

    # 입력 소스 유형 판별 및 처리
    if source_path and source_path.is_file():
        # 단일 이미지 파일
        print(f"\n[모드] 단일 이미지: {source}")
        process_image(model, source_path, args, output_dir)

    elif source_path and source_path.is_dir():
        # 폴더
        print(f"\n[모드] 폴더: {source}")
        process_folder(model, source_path, args, output_dir)

    elif source.startswith(('rtsp://', 'http://', 'https://')) or source.isdigit():
        # 스트림 (RTSP/HTTP/웹캠)
        print(f"\n[모드] 스트림: {source}")
        if not args.show and not args.save:
            args.show = True  # 스트림 모드에서는 기본적으로 화면 표시
        process_stream(model, source, args, output_dir)

    else:
        print(f"[ERROR] 유효하지 않은 소스: {source}")
        print("사용법:")
        print("  이미지: python inference.py --source image.jpg")
        print("  폴더:   python inference.py --source ./images/")
        print("  RTSP:   python inference.py --source rtsp://ip:port/stream")
        print("  웹캠:   python inference.py --source 0")

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
