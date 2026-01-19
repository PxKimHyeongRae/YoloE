# PPE + Fire + Fall 통합 데이터셋 분석 보고서

## 목표
단일 YOLOE 모델로 **PPE(헬멧/안전조끼)**, **화재(불꽃/연기)**, **쓰러짐**을 동시에 탐지

---

## 타겟 클래스 (7개)

| ID | 클래스명 | 설명 | 객체 수 | 비율 |
|----|----------|------|---------|------|
| 0 | Helmet_OFF | 헬멧 미착용 | 9,128 | 8.4% |
| 1 | Helmet_ON | 헬멧 착용 | 18,647 | 17.1% |
| 2 | Vest_OFF | 안전조끼 미착용 | 10,183 | 9.3% |
| 3 | Vest_ON | 안전조끼 착용 | 16,768 | 15.4% |
| 4 | fire | 화재/불꽃 | 30,097 | 27.6% |
| 5 | smoke | 연기 | 11,616 | 10.7% |
| 6 | fall | 쓰러짐/넘어짐 | 12,494 | 11.5% |

---

## 통합 데이터셋 규모

| Split | 이미지 수 | 비율 |
|-------|-----------|------|
| Train | 48,294 | 84.9% |
| Valid | 5,606 | 9.8% |
| Test | 3,018 | 5.3% |
| **Total** | **56,918** | 100% |

**총 객체 수: 108,933개** (이미지당 평균 1.91개)

---

## 사용된 원본 데이터셋

### PPE 데이터셋 (7개)

| 데이터셋 | 원본 이미지 | 샘플링 | Helmet | Vest | 비고 |
|----------|-------------|--------|--------|------|------|
| construction safety.v2 | 1,206 | 100% | ✓ | ✓ | 헬멧+조끼 모두 포함 |
| ppe_only_700 | 1,077 | 100% | ✓ | ✓ | 4클래스 완전 라벨링 |
| ppe_vest_13k | 14,809 | 30% | - | ✓ | 조끼만 (대용량) |
| Safety vest.v3i | 6,710 | 30% | - | ✓ | 조끼만 |
| PPE-0.3.v1i.yolo26 | 8,725 | 50% | ✓ | ✓ | 16클래스 중 4개 사용 |
| Hard hat.v1i.yolo26 | 14,544 | 30% | ✓ | - | 헬멧만 (대용량) |
| PPE.v14-allinone.yolo26 | 4,067 | 100% | ✓ | ✓ | 9클래스 중 4개 사용 |

### Fire 데이터셋 (3개)

| 데이터셋 | 원본 이미지 | 샘플링 | fire | smoke | 비고 |
|----------|-------------|--------|------|-------|------|
| fire.v2i.yolo26 | 4,504 | 100% | ✓ | ✓ | 기본 화재 데이터 |
| Fire.v2i.yolo26 (1) | 4,504 | 100% | ✓ | ✓ | 추가 화재 데이터 |
| Fire-Final.v1-mitesh.yolo26 | 28,937 | 50% | ✓ | ✓ | 대용량 (PDPU 제외) |

### Fall 데이터셋 (3개)

| 데이터셋 | 원본 이미지 | 샘플링 | 원본 클래스 | 비고 |
|----------|-------------|--------|-------------|------|
| Fall.v1i.yolo26 | ~4,000 | 100% | down(1) | 쓰러진 상태 |
| fall.v2i.yolo26 | ~4,000 | 100% | fall(0) | 넘어지는 동작 |
| fall.v4i.yolo26 | ~4,000 | 100% | falling(0) | 넘어지는 동작 |

---

## 클래스 매핑 상세

### PPE 매핑
```
# construction safety.v2-release.yolo26_helmet_no-helmet_no-vest_vest_person
helmet(0) → Helmet_ON(1)
no-helmet(1) → Helmet_OFF(0)
no-vest(2) → Vest_OFF(2)
vest(4) → Vest_ON(3)
# person(3) 제외

# ppe_only_700
Hard_Hat_OFF(0) → Helmet_OFF(0)
Hard_Hat_ON(1) → Helmet_ON(1)
Safety_Vest_OFF(2) → Vest_OFF(2)
Safety_Vest_ON(3) → Vest_ON(3)

# ppe_vest_13k
no_safety_vest(0) → Vest_OFF(2)
safety_vest(1) → Vest_ON(3)

# Safety vest.v3i.yolo26_vest_no-vest_6k
NO-Safety Vest(0) → Vest_OFF(2)
Safety Vest(1) → Vest_ON(3)

# PPE-0.3.v1i.yolo26 (16클래스 중 4개만 사용)
Helmet(3) → Helmet_ON(1)
Without Helmet(12) → Helmet_OFF(0)
Vest(8) → Vest_ON(3)
Without Vest(15) → Vest_OFF(2)

# Hard hat.v1i.yolo26
0 hard hat(0) → Helmet_ON(1)
1 no(1) → Helmet_OFF(0)

# PPE.v14-allinone.yolo26 (9클래스 중 4개만 사용)
hardhat(2) → Helmet_ON(1)
no_hardhat(5) → Helmet_OFF(0)
vest(8) → Vest_ON(3)
no_vest(6) → Vest_OFF(2)
```

### Fire 매핑
```
# fire.v2i.yolo26
fire(0) → fire(4)
smoke(1) → smoke(5)

# Fire.v2i.yolo26 (1)
fire(0) → fire(4)
smoke(1) → smoke(5)

# Fire-Final.v1-mitesh.yolo26 (3클래스 중 2개만 사용)
fire(1) → fire(4)
smoke(2) → smoke(5)
# PDPU(0) 제외
```

### Fall 매핑
```
# Fall.v1i.yolo26 (3클래스 중 1개만 사용)
down(1) → fall(6)
# 10-(0), person(2) 제외

# fall.v2i.yolo26 (3클래스 중 1개만 사용)
fall(0) → fall(6)
# sit(1), walk(2) 제외

# fall.v4i.yolo26 (6클래스 중 1개만 사용)
falling(0) → fall(6)
# sitting(1), sleeping(2), standing(3), walking(4), waving hands(5) 제외
```

---

## 학습 방법

### 필수 요구사항
- **YOLOE 모델** + **YOLOEPETrainer** 사용 필수
- 세그멘테이션 사전학습 가중치 (`*-seg.pt`) 사용

### 사용 가능한 모델
| 모델 | 파라미터 | 권장 용도 |
|------|----------|-----------|
| yoloe-26n | ~2.5M | 엣지 디바이스, 실시간 |
| yoloe-26s | ~7M | 모바일, 경량 서버 |
| yoloe-26m | ~20M | 일반 서버 (권장) |
| yoloe-26l | ~43M | 고성능 서버 |
| yoloe-26x | ~68M | 최고 정확도 |

### 실행 명령어
```bash
cd C:/task/yoloe
python train_unified.py                    # 기본 (yoloe-26m, 100 epochs)
python train_unified.py --model yoloe-26s  # 작은 모델
python train_unified.py --model yoloe-26l  # 큰 모델
python train_unified.py --epochs 200       # 에폭 변경
python train_unified.py --batch 8          # 배치 크기 조정 (GPU 메모리 부족 시)
```

### 학습 설정 (기본값)
```python
{
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "patience": 30,        # Early stopping
    "mosaic": 1.0,
    "mixup": 0.1,
}
```

---

## 파일 구조

```
C:/task/yoloe/
├── train_unified.py                  # ⭐ 통합 학습 스크립트
├── train_fire_yoloe.py               # Fire 전용 학습 (선택)
├── train_vest.py                     # Vest 전용 학습 (참고용)
│
├── models/                           # 사전학습 가중치
│   ├── yoloe-26n-seg.pt
│   ├── yoloe-26s-seg.pt
│   ├── yoloe-26m-seg.pt
│   ├── yoloe-26l-seg.pt
│   └── yoloe-26x-seg.pt
│
├── dataset/
│   ├── DATA_ANALYSIS.md              # 이 문서
│   ├── build_unified_all_dataset.py  # 통합 데이터셋 생성 스크립트
│   ├── build_fire_dataset.py         # Fire 데이터셋 생성 스크립트
│   │
│   ├── unified_all/                  # ⭐ 통합 데이터셋 (PPE + Fire + Fall)
│   │   ├── data.yaml
│   │   ├── train/images/ (48,294장)
│   │   ├── train/labels/
│   │   ├── valid/images/ (5,606장)
│   │   ├── valid/labels/
│   │   ├── test/images/ (3,018장)
│   │   └── test/labels/
│   │
│   ├── fire_smoke_unified/           # Fire 전용 데이터셋
│   │
│   └── [원본 데이터셋들]/
│       ├── construction safety.v2.../
│       ├── ppe_only_700.../
│       ├── ppe_vest_13k/
│       ├── Safety vest.v3i.../
│       ├── PPE-0.3.v1i.yolo26/
│       ├── Hard hat.v1i.yolo26/
│       ├── PPE.v14-allinone.yolo26/
│       ├── fire.v2i.yolo26/
│       ├── Fire.v2i.yolo26 (1)/
│       ├── Fire-Final.v1-mitesh.yolo26/
│       ├── Fall.v1i.yolo26/
│       ├── fall.v2i.yolo26/
│       └── fall.v4i.yolo26/
│
└── runs/                             # 학습 결과 저장
    └── unified/
        └── yoloe-26m_e100_b16/
            ├── weights/
            │   ├── best.pt           # 최고 성능 모델
            │   └── last.pt           # 마지막 체크포인트
            └── [학습 로그 및 그래프]
```

---

## 클래스 균형 분석

| 카테고리 | 클래스 | 객체 수 | 전체 비율 | 카테고리 내 비율 |
|----------|--------|---------|-----------|------------------|
| **PPE** | Helmet_OFF | 9,128 | 8.4% | 16.7% |
| **PPE** | Helmet_ON | 18,647 | 17.1% | 34.1% |
| **PPE** | Vest_OFF | 10,183 | 9.3% | 18.6% |
| **PPE** | Vest_ON | 16,768 | 15.4% | 30.6% |
| | **PPE 소계** | **54,726** | **50.2%** | 100% |
| **Fire** | fire | 30,097 | 27.6% | 72.2% |
| **Fire** | smoke | 11,616 | 10.7% | 27.8% |
| | **Fire 소계** | **41,713** | **38.3%** | 100% |
| **Fall** | fall | 12,494 | 11.5% | 100% |
| | **Fall 소계** | **12,494** | **11.5%** | 100% |

**총계: 108,933개 객체**

### 클래스 불균형 참고사항
- `fire` 클래스가 가장 많음 (27.6%) - Fire-Final 데이터셋이 대용량
- `Helmet_OFF`가 가장 적음 (8.4%) - 실제 현장에서 미착용이 적음
- 학습 시 class weight 조정 또는 focal loss 고려 가능

---

## 데이터셋 재생성

통합 데이터셋을 다시 생성해야 할 경우:

```bash
cd C:/task/yoloe/dataset
python build_unified_all_dataset.py
```

이 스크립트는:
1. 모든 원본 데이터셋에서 이미지와 라벨 수집
2. 클래스 ID를 통합 ID(0-6)로 매핑
3. 샘플링 비율 적용 (대용량 데이터셋)
4. train/valid/test 폴더로 분류
5. data.yaml 자동 생성

---

## 버전 정보

| 항목 | 값 |
|------|-----|
| 문서 생성일 | 2025-01-19 |
| 최종 업데이트 | 2025-01-19 |
| 이미지 총 수 | 56,918장 |
| 객체 총 수 | 108,933개 |
| 클래스 수 | 7개 |
| 학습 모델 | YOLOE (yoloe-26n/s/m/l/x) |
| 학습 방식 | YOLOEPETrainer + 세그멘테이션 가중치 |

---

## 예상 결과물

학습 완료 후 생성되는 파일:
- `runs/unified/[실험명]/weights/best.pt` - 최고 성능 모델
- `runs/unified/[실험명]/weights/last.pt` - 마지막 체크포인트

### 추론 사용 예시
```python
from ultralytics import YOLOE

model = YOLOE("runs/unified/yoloe-26m_e100_b16/weights/best.pt")
results = model.predict("image.jpg", conf=0.5)

# 클래스명
# 0: Helmet_OFF, 1: Helmet_ON, 2: Vest_OFF, 3: Vest_ON
# 4: fire, 5: smoke, 6: fall
```
