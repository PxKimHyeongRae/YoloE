# PPE + Fire + Fall 통합 데이터셋 분석 보고서

## 목표
단일 YOLOE 모델로 **PPE(헬멧/안전조끼)**, **화재(불꽃/연기)**, **쓰러짐**을 동시에 탐지

---

## 타겟 클래스 (8개)

| ID | 클래스명 | 설명 | 객체 수 | 비율 |
|----|----------|------|---------|------|
| 0 | Helmet_OFF | 헬멧 미착용 | 9,128 | 6.6% |
| 1 | Helmet_ON | 헬멧 착용 | 18,647 | 13.5% |
| 2 | Vest_OFF | 안전조끼 미착용 | 10,183 | 7.4% |
| 3 | Vest_ON | 안전조끼 착용 | 16,768 | 12.1% |
| 4 | fire | 화재/불꽃 | 30,097 | 21.8% |
| 5 | smoke | 연기 | 11,616 | 8.4% |
| 6 | fall | 쓰러짐/넘어짐 | 14,804 | 10.7% |
| 7 | person | 정상 자세 (서있음/앉음/걸음) | 29,108 | 21.0% |

**총 객체 수: 140,351개**

---

## 통합 데이터셋 규모

| Split | 이미지 수 | 비율 |
|-------|-----------|------|
| Train | 55,319 | 86.0% |
| Valid | 5,873 | 9.1% |
| Test | 3,110 | 4.8% |
| **Total** | **64,302** | 100% |

---

## 클래스 설계 원칙

### PPE (0-3): ON/OFF 쌍으로 학습
```
Helmet_OFF ↔ Helmet_ON  (같은 영역에서 있음/없음)
Vest_OFF ↔ Vest_ON      (같은 영역에서 있음/없음)
→ ON을 알아야 OFF를 정확히 구분
```

### Fire (4-5): 위험 상황 자체가 탐지 대상
```
fire, smoke → 있으면 탐지, 없으면 정상
→ 별도 정상 클래스 불필요
```

### Fall (6-7): 정상/비정상 자세 구분
```
fall ↔ person  (같은 "사람"의 다른 상태)
→ 정상 자세(person)를 알아야 비정상(fall)을 구분
```

---

## 클래스 매핑 상세

### PPE 매핑
```
# construction safety.v2
helmet(0) → Helmet_ON(1), no-helmet(1) → Helmet_OFF(0)
no-vest(2) → Vest_OFF(2), vest(4) → Vest_ON(3)

# ppe_only_700
Hard_Hat_OFF(0) → Helmet_OFF(0), Hard_Hat_ON(1) → Helmet_ON(1)
Safety_Vest_OFF(2) → Vest_OFF(2), Safety_Vest_ON(3) → Vest_ON(3)

# ppe_vest_13k
no_safety_vest(0) → Vest_OFF(2), safety_vest(1) → Vest_ON(3)

# PPE-0.3.v1i.yolo26
Helmet(3) → Helmet_ON(1), Without Helmet(12) → Helmet_OFF(0)
Vest(8) → Vest_ON(3), Without Vest(15) → Vest_OFF(2)

# Hard hat.v1i.yolo26
0 hard hat(0) → Helmet_ON(1), 1 no(1) → Helmet_OFF(0)

# PPE.v14-allinone.yolo26
hardhat(2) → Helmet_ON(1), no_hardhat(5) → Helmet_OFF(0)
vest(8) → Vest_ON(3), no_vest(6) → Vest_OFF(2)
```

### Fire 매핑
```
# fire.v2i.yolo26 / Fire.v2i.yolo26 (1)
fire(0) → fire(4), smoke(1) → smoke(5)

# Fire-Final.v1-mitesh.yolo26
fire(1) → fire(4), smoke(2) → smoke(5)
# PDPU(0) 제외 (의미 불명)
```

### Fall 매핑 (v2 - person 클래스 추가)
```
# Fall.v1i.yolo26
down(1) → fall(6)
person(2) → person(7)
# 10-(0) 제외 (의미 불명)

# fall.v2i.yolo26
fall(0) → fall(6)
sit(1) → person(7)
walk(2) → person(7)

# fall.v4i.yolo26
falling(0) → fall(6)
sleeping(2) → fall(6)  ← 누워있는 상태는 위험으로 간주
sitting(1) → person(7)
standing(3) → person(7)
walking(4) → person(7)
# waving hands(5) 제외 (소량, 관련 없음)
```

---

## 학습 방법

### 실행 명령어
```bash
cd C:/task/yoloe
python train_unified.py                    # 기본 (yoloe-26m, 100 epochs)
python train_unified.py --model yoloe-26s  # 작은 모델
python train_unified.py --model yoloe-26l  # 큰 모델
python train_unified.py --epochs 200       # 에폭 변경
python train_unified.py --batch 8          # 배치 크기 조정
```

### 사용 가능한 모델
| 모델 | 파라미터 | 권장 용도 |
|------|----------|-----------|
| yoloe-26n | ~2.5M | 엣지 디바이스, 실시간 |
| yoloe-26s | ~7M | 모바일, 경량 서버 |
| yoloe-26m | ~20M | 일반 서버 (권장) |
| yoloe-26l | ~43M | 고성능 서버 |
| yoloe-26x | ~68M | 최고 정확도 |

---

## 실제 서비스 적용

### 알림 발생 조건 (추론 시 필터링)
```python
# 위험 상황만 알림
ALERT_CLASSES = {
    0: "헬멧 미착용 감지",
    2: "안전조끼 미착용 감지",
    4: "화재 감지",
    5: "연기 감지 (화재 주의)",
    6: "쓰러짐 감지",
}

# 학습에는 사용하지만 알림 안 함
# 1: Helmet_ON, 3: Vest_ON, 7: person
```

### 추론 예시
```python
from ultralytics import YOLOE

model = YOLOE("runs/unified/best.pt")
results = model.predict("image.jpg", conf=0.5)

for box in results[0].boxes:
    cls = int(box.cls)
    if cls in [0, 2, 4, 5, 6]:  # 위험 클래스만
        # 알림 발생
        pass
```

---

## 파일 구조

```
C:/task/yoloe/
├── train_unified.py                  # 통합 학습 스크립트
├── dataset/
│   ├── DATA_ANALYSIS.md              # 이 문서
│   ├── build_unified_all_dataset.py  # 데이터셋 생성 스크립트
│   └── unified_all/                  # 통합 데이터셋
│       ├── data.yaml
│       ├── train/ (55,319장)
│       ├── valid/ (5,873장)
│       └── test/ (3,110장)
└── models/                           # 사전학습 가중치
    └── yoloe-26*-seg.pt
```

---

## 버전 정보

| 항목 | 값 |
|------|-----|
| 버전 | v2 (person 클래스 추가) |
| 최종 업데이트 | 2025-01-19 |
| 이미지 총 수 | 64,302장 |
| 객체 총 수 | 140,351개 |
| 클래스 수 | 8개 |
| 학습 모델 | YOLOE + YOLOEPETrainer |
