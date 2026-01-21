# YOLOE Safety Detection Project

## 프로젝트 개요

산업 현장 안전 감지를 위한 YOLOE 기반 객체 검출 모델 학습 프로젝트.
헬멧 착용 여부, 화재, 연기, 쓰러진 사람 등을 실시간으로 검출.

## 핵심 정보

### 현재 클래스 (6개)
```yaml
0: 'person without helmet'  # 헬멧 미착용
1: 'person wearing helmet'  # 헬멧 착용
2: 'fire'                   # 화재
3: 'smoke'                  # 연기
4: 'fallen person'          # 쓰러진 사람
5: 'person'                 # 일반 사람
```

### 주요 데이터셋
- `dataset/unified_no_vest/` - 현재 사용 중인 통합 데이터셋 (6클래스, ~66k 이미지)
- `dataset/hard_hat_cleaned/` - 정제된 헬멧 데이터
- `dataset/fire_smoke_unified/` - 화재/연기 데이터

### 사전학습 모델
- `models/yoloe-26l-seg.pt` - YOLOE Large (권장)
- `models/yoloe-26m-seg.pt` - YOLOE Medium
- `models/yoloe-26s-seg.pt` - YOLOE Small

### 학습된 모델
- `best.pt`, `best_yoloe26m_unifed.pt` - 통합 모델 (8클래스, Vest 포함)

## 주요 스크립트

### 학습
```bash
# Linear Probing (빠른 검증, 10 epochs)
python train_unified.py --mode linear --model yoloe-26l

# Full Tuning (성능 최적화, 80 epochs)
python train_unified.py --mode full --model yoloe-26l

# 기존 가중치에서 계속 학습
python train_unified.py --mode full --weights runs/unified/exp/weights/best.pt
```

### 추론
```bash
# 이미지/폴더 추론
python inference.py --source image.jpg
python inference.py --source folder/ --sample 100 --random

# RTSP 스트림
python inference.py --source rtsp://ip:port/stream
```

### 데이터셋 빌드
```bash
# 통합 데이터셋 생성 (Vest 제외, 6클래스)
python dataset/build_unified_no_vest.py
```

## YOLOE 특이사항

### Resume 학습 불가
YOLOE는 optimizer 상태 복원 시 에러 발생. 대안:
- `best.pt` 가중치만 로드 후 새 학습 시작
- 처음부터 충분한 epochs + patience 설정

### 클래스 네이밍
YOLOE는 CLIP 텍스트 인코더 사용. 자연어 클래스명 권장:
- Bad: `Helmet_OFF`, `Helmet_ON`
- Good: `person without helmet`, `person wearing helmet`

### Trainer 클래스
```python
from ultralytics.models.yolo.yoloe import YOLOEPETrainer
model.train(trainer=YOLOEPETrainer, ...)
```

## 프로젝트 구조

```
yoloe/
├── train_unified.py      # 메인 학습 스크립트
├── inference.py          # 추론 스크립트
├── dataset/
│   ├── unified_no_vest/  # 현재 데이터셋 (6클래스)
│   │   └── data.yaml
│   ├── build_unified_no_vest.py
│   └── ...
├── models/               # 사전학습 가중치
│   └── yoloe-26{s,m,l,x}-seg.pt
├── docs/
│   └── yoloe_training_guide.md  # 상세 가이드
└── runs/                 # 학습 결과
```

## 관련 프로젝트

- `C:\task\vision\video_analytics\` - RTSP 기반 실시간 영상 분석 시스템
  - `core/detector.py` - YOLOEDetector 클래스
  - `config.py` - 모델 경로, exclude_classes 설정

## 참고 문서

- `docs/yoloe_training_guide.md` - YOLOE 학습 완벽 가이드 (~1300줄)
- `docs/experiments.md` - 실험 기록
- `dataset/README.md` - 데이터셋 설명

## 주의사항

1. **리눅스 서버와 동기화 필요** - Windows에서 수정한 파일은 리눅스 서버로 복사 필요
2. **Vest 클래스 제거됨** - 성능 문제로 Vest_OFF, Vest_ON 클래스 제외
3. **fire, smoke 사전학습 부족** - Objects365/LVIS에 없어서 fine-tuning 필수
