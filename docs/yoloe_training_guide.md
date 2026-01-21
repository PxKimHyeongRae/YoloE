# YOLOE 학습 방법 및 클래스 네이밍 전략 완벽 가이드

> 조사일: 2026-01-21
> YOLOE (Real-Time Seeing Anything) - ICCV 2025
> Ultralytics 통합 버전 기준

---

## 목차

1. [YOLOE 아키텍처 심층 분석](#1-yoloe-아키텍처-심층-분석)
2. [YOLOE vs YOLO-World 비교](#2-yoloe-vs-yolo-world-비교)
3. [사전학습 데이터셋 상세](#3-사전학습-데이터셋-상세)
4. [Fine-tuning 전략](#4-fine-tuning-전략)
5. [클래스 네이밍 전략](#5-클래스-네이밍-전략)
6. [정확도 향상 방법론](#6-정확도-향상-방법론)
7. [Resume 학습 문제와 해결](#7-resume-학습-문제와-해결)
8. [실전 사용법](#8-실전-사용법)
9. [트러블슈팅](#9-트러블슈팅)
10. [참고 자료](#10-참고-자료)

---

## 1. YOLOE 아키텍처 심층 분석

### 1.1 개요

YOLOE는 "Real-Time Seeing Anything"을 목표로 하는 open-vocabulary 객체 검출 및 세그멘테이션 모델입니다. 기존 YOLO 시리즈와 달리 **고정된 클래스에 제한되지 않고**, 텍스트, 이미지, 또는 내장 어휘를 통해 어떤 객체든 검출할 수 있습니다.

**핵심 특징:**
- Zero-shot 검출 능력 (학습하지 않은 클래스도 검출)
- 3가지 프롬프트 방식 지원 (Text / Visual / Prompt-free)
- Re-parameterization으로 추론 시 오버헤드 제로
- YOLOv8/v11 기반 아키텍처

### 1.2 기존 YOLO와의 차이점

| 항목 | 기존 YOLO (v5/v8/v11) | YOLOE |
|------|----------------------|-------|
| 클래스 정의 | 학습 시 고정 | 추론 시 동적 변경 가능 |
| 텍스트 이해 | 불가능 | MobileCLIP-B(LT) 인코더 |
| 프롬프트 | 없음 | Text / Visual / Prompt-free |
| 사전학습 | COCO (80 클래스) | Objects365 + LVIS + GoldG (1200+ 클래스) |
| Zero-shot | 불가능 | 가능 |
| 세그멘테이션 | 별도 모델 필요 | 통합 지원 |
| Re-parameterization | 해당 없음 | 학습 후 일반 YOLO로 변환 |

### 1.3 YOLOE의 3가지 핵심 모듈

#### 1.3.1 RepRTA (Re-parameterizable Region-Text Alignment)

**목적:** 텍스트 프롬프트를 효율적으로 처리

**동작 원리:**
1. MobileCLIP-B(LT) 텍스트 인코더로 클래스명 임베딩 생성
2. 임베딩을 학습 전 캐싱 (인코더 오버헤드 제거)
3. SwiGLU feed-forward 보조 네트워크로 임베딩 정제
4. **학습 후 re-parameterization**: 보조 네트워크 가중치를 classification head에 융합

**Re-parameterization 수식:**
```
W_final = W_cls + W_aux @ W_text_embedding
```

**장점:**
- 학습 시에만 보조 네트워크 사용
- 추론 시 일반 YOLO와 동일한 구조 → 속도 저하 없음
- 텍스트 인코더 없이 추론 가능

#### 1.3.2 SAVPE (Semantic-Activated Visual Prompt Encoder)

**목적:** 이미지 기반 프롬프트 처리 (Visual Prompt)

**구조:**
```
┌─────────────────────────────────────────────┐
│                   SAVPE                      │
├─────────────────┬───────────────────────────┤
│ Semantic Branch │ Activation Branch          │
│ (D channels)    │ (A=16 groups)              │
│                 │                            │
│ 프롬프트 무관    │ 프롬프트 인식              │
│ 시맨틱 특징     │ 그룹별 가중치 생성          │
└─────────────────┴───────────────────────────┘
                    ↓
              Aggregation
                    ↓
           Visual Prompt Embedding
```

**사용법:**
```python
# 클래스당 N개 (기본 16장)의 참조 이미지로 visual embedding 생성
reference_images = {
    "fire": [img1, img2, ..., img16],
    "smoke": [img1, img2, ..., img16],
}
visual_pe = model.get_visual_pe(reference_images)
model.set_classes(class_names, visual_pe)
```

**장점:**
- 텍스트로 표현하기 어려운 객체에 효과적
- 도메인 특화 시각적 특징 직접 인코딩
- 텍스트 프롬프트보다 정확할 수 있음

#### 1.3.3 LRPC (Lazy Region-Prompt Contrast)

**목적:** Prompt-free 모드에서 객체 인식

**동작 원리:**
1. 특수한 "all-object" 프롬프트 임베딩 학습
2. 이 임베딩으로 모든 객체를 하나의 카테고리로 검출
3. 검출된 영역에 대해서만 내장 어휘(4585개)에서 클래스 검색
4. 코사인 유사도로 가장 유사한 클래스 할당

**내장 어휘:**
- RAM++ (Recognize Anything Model Plus) 태그셋 기반
- 4585개 고유 태그 ID
- 6449개 원본 태그 (동의어 그룹화 후 4585개)

**장점:**
- 외부 언어 모델 불필요
- 1.7배 빠른 추론 (직접 어휘 매칭 대비)
- 대규모 어휘에서도 효율적

### 1.4 학습 과정 (3단계)

```
┌────────────────────────────────────────────────────────────┐
│ Stage 1: Text Prompt Training (30 epochs)                  │
│ - 데이터: Objects365 + GoldG                               │
│ - 목적: RepRTA 학습, region-text alignment                │
│ - 전체 모델 학습                                           │
├────────────────────────────────────────────────────────────┤
│ Stage 2: Visual Prompt Training (2 epochs)                 │
│ - SAVPE 인코더만 학습                                      │
│ - 다른 파라미터 동결                                       │
├────────────────────────────────────────────────────────────┤
│ Stage 3: Prompt-free Training (1 epoch)                    │
│ - Specialized prompt embedding만 학습                      │
│ - LRPC 모듈 활성화                                         │
└────────────────────────────────────────────────────────────┘

총 학습 비용: YOLO-World 대비 3배 적음
```

### 1.5 모델 변형

| 모델 | 백본 | 파라미터 | FLOPs | LVIS AP | 속도 |
|------|------|----------|-------|---------|------|
| YOLOE-v8-S | YOLOv8-S | 11.4M | 28.8G | 27.9 | 빠름 |
| YOLOE-v8-M | YOLOv8-M | 25.9M | 79.3G | 32.6 | 중간 |
| YOLOE-v8-L | YOLOv8-L | 43.7M | 167.4G | 35.9 | 느림 |
| YOLOE-11-S | YOLO11-S | 9.5M | 21.6G | 27.5 | 빠름 |
| YOLOE-11-M | YOLO11-M | 20.1M | 68.0G | 32.1 | 중간 |
| YOLOE-11-L | YOLO11-L | 25.4M | 87.0G | 34.2 | 느림 |

**Ultralytics 명명 규칙:**
- `yoloe-26s` = YOLOE-v8-S (또는 YOLOE-11-S)
- `yoloe-26m` = YOLOE-v8-M
- `yoloe-26l` = YOLOE-v8-L
- `-seg` 접미사: 세그멘테이션 가중치 포함

---

## 2. YOLOE vs YOLO-World 비교

### 2.1 발전 과정

```
CLIP (OpenAI)
    ↓
YOLO-World (CVPR 2024) - 최초의 실시간 open-vocabulary YOLO
    ↓
YOLOE (ICCV 2025) - 개선된 효율성과 다중 프롬프트 지원
```

### 2.2 상세 비교

| 항목 | YOLO-World | YOLOE |
|------|------------|-------|
| 텍스트 인코더 | CLIP ViT-B/32 | MobileCLIP-B(LT) |
| 추론 시 인코더 | 필요 (또는 re-param) | 불필요 (re-param 기본) |
| Visual Prompt | 미지원 | SAVPE로 지원 |
| Prompt-free | 미지원 | LRPC로 지원 |
| 학습 비용 | 높음 | 1/3 수준 |
| LVIS AP (L 모델) | 32.4 | 35.9 (+3.5) |
| 추론 속도 | 52.0 FPS | 72.8 FPS (1.4배) |
| 세그멘테이션 | 별도 | 통합 |

### 2.3 YOLO-World의 핵심 기술 (YOLOE에 계승)

**RepVL-PAN (Re-parameterizable Vision-Language PAN):**
- Vision과 Language 특징을 연결하는 Path Aggregation Network
- 학습 시: Cross-attention으로 텍스트-이미지 상호작용
- 추론 시: Re-parameterization으로 일반 PAN으로 변환

**Region-Text Contrastive Loss:**
```python
# 의사 코드
loss = contrastive_loss(
    region_features,      # 검출된 영역의 시각적 특징
    text_embeddings,      # 클래스명의 텍스트 임베딩
    temperature=0.07
)
```

---

## 3. 사전학습 데이터셋 상세

### 3.1 Objects365 (365개 클래스)

**개요:**
- Megvii 연구팀 제작
- 200만 이미지, 3000만+ 바운딩 박스
- 일반 객체 365개 카테고리

**전체 클래스 목록 (관련 클래스 강조):**

```
Person ✅, Sneakers, Chair, Other Shoes, Hat, Car, Lamp, Glasses, Bottle,
Desk, Cup, Street Lights, Cabinet/shelf, Handbag/Satchel, Bracelet, Plate,
Picture/Frame, Helmet ✅, Book, Gloves, Storage box, Boat, Leather Shoes,
Flower, Bench, Potted Plant, Bowl/Basin, Flag, Pillow, Boots, Vase,
Microphone, Necklace, Ring, SUV, Wine Glass, Belt, Monitor/TV, Backpack,
Umbrella, Traffic Light, Speaker, Watch, Tie, Trash bin Can, Slippers,
Bicycle, Stool, Barrel/bucket, Van, Couch, Sandals, Basket, Drum, Pen/Pencil,
Bus, Wild Bird, High Heels, Motorcycle, Guitar, Carpet, Cell Phone, Bread,
Camera, Canned, Truck, Traffic cone, Cymbal, Lifesaver, Towel, Stuffed Toy,
Candle, Sailboat, Laptop, Awning, Bed, Faucet, Tent, Horse, Mirror,
Power outlet, Sink, Apple, Air Conditioner, Knife, Hockey Stick, Paddle,
Pickup Truck, Fork, Traffic Sign, Balloon, Tripod, Dog, Spoon, Clock, Pot,
Cow, Cake, Dinning Table, Sheep, Hanger, Blackboard/Whiteboard, Napkin,
Other Fish, Orange/Tangerine, Toiletry, Keyboard, Tomato, Lantern,
Machinery Vehicle, Fan, Green Vegetables, Banana, Baseball Glove, Airplane,
Mouse, Train, Pumpkin, Soccer, Skiboard, Luggage, Nightstand, Tea pot,
Telephone, Trolley, Head Phone, Sports Car, Stop Sign, Dessert, Scooter,
Stroller, Crane, Remote, Refrigerator, Oven, Lemon, Duck, Baseball Bat,
Surveillance Camera, Cat, Jug, Broccoli, Piano, Pizza, Elephant, Skateboard,
Surfboard, Gun, Skating and Skiing shoes, Gas stove, Donut, Bow Tie, Carrot,
Toilet, Kite, Strawberry, Other Balls, Shovel, Pepper, Computer Box,
Toilet Paper, Cleaning Products, Chopsticks, Microwave, Pigeon, Baseball,
Cutting/chopping Board, Coffee Table, Side Table, Scissors, Marker, Pie,
Ladder, Snowboard, Cookies, Radiator, Fire Hydrant ⚠️, Basketball, Zebra,
Grape, Giraffe, Potato, Sausage, Tricycle, Violin, Egg, Fire Extinguisher ⚠️,
Candy, Fire Truck ⚠️, Billiards, Converter, Bathtub, Wheelchair, Golf Club,
Briefcase, Cucumber, Cigar/Cigarette, Paint Brush, Pear, Heavy Truck,
Hamburger, Extractor, Extension Cord, Tong, Tennis Racket, Folder,
American Football, earphone, Mask, Kettle, Tennis, Ship, Swing,
Coffee Machine, Slide, Carriage, Onion, Green beans, Projector, Frisbee,
Washing Machine/Drying Machine, Chicken, Printer, Watermelon, Saxophone,
Tissue, Toothbrush, Ice cream, Hot-air balloon, Cello, French Fries, Scale,
Trophy, Cabbage, Hot dog, Blender, Peach, Rice, Wallet/Purse, Volleyball,
Deer, Goose, Tape, Tablet, Cosmetics, Trumpet, Pineapple, Golf Ball,
Ambulance, Parking meter, Mango, Key, Hurdle, Fishing Rod, Medal, Flute,
Brush, Penguin, Megaphone, Corn, Lettuce, Garlic, Swan, Helicopter,
Green Onion, Sandwich, Nuts, Speed Limit Sign, Induction Cooker, Broom,
Trombone, Plum, Rickshaw, Goldfish, Kiwi fruit, Router/modem, Poker Card,
Toaster, Shrimp, Sushi, Cheese, Notepaper, Cherry, Pliers, CD, Pasta,
Hammer, Cue, Avocado, Hamimelon, Flask, Mushroom, Screwdriver, Soap,
Recorder, Bear, Eggplant, Board Eraser, Coconut, Tape Measure/Ruler, Pig,
Showerhead, Globe, Chips, Steak, Crosswalk Sign, Stapler, Camel, Formula 1,
Pomegranate, Dishwasher, Crab, Hoverboard, Meat ball, Rice Cooker, Tuba,
Calculator, Papaya, Antelope, Parrot, Seal, Butterfly, Dumbbell, Donkey,
Lion, Urinal, Dolphin, Electric Drill, Hair Dryer, Egg tart, Jellyfish,
Treadmill, Lighter, Grapefruit, Game board, Mop, Radish, Baozi, Target,
French, Spring Rolls, Monkey, Rabbit, Pencil Case, Yak, Red Cabbage,
Binoculars, Asparagus, Barbell, Scallop, Noddles, Comb, Dumpling, Oyster,
Table Tennis paddle, Cosmetics Brush/Eyeliner Pencil, Chainsaw, Eraser,
Lobster, Durian, Okra, Lipstick, Cosmetics Mirror, Curling, Table Tennis
```

**우리 프로젝트 관련:**
- ✅ Person: 있음
- ✅ Helmet: 있음
- ⚠️ Fire: 없음 (Fire Hydrant, Fire Extinguisher, Fire Truck만 있음)
- ❌ Smoke: 없음
- ❌ Fall/Fallen: 없음 (행동/상태)

### 3.2 LVIS (1203개 클래스)

**개요:**
- Facebook AI Research (FAIR) 제작
- Large Vocabulary Instance Segmentation
- COCO 이미지 기반, 더 세분화된 어노테이션
- Long-tail 분포 (희귀 클래스 포함)

**클래스 분포:**
- Frequent (빈번): ~400개
- Common (보통): ~400개
- Rare (희귀): ~400개

**주요 관련 클래스:**
```
person ✅
fire_alarm, fire_extinguisher, fire_hose, fireplace ⚠️ (불꽃 자체는 없음)
helmet ❌ (없음!)
smoke ❌
```

**LVIS vs COCO:**
| 항목 | COCO | LVIS |
|------|------|------|
| 클래스 수 | 80 | 1203 |
| 이미지 | 118K | 164K |
| 인스턴스 | 860K | 2M |
| 세분화 | 낮음 | 높음 |

### 3.3 GoldG (Grounding Data)

**구성:**
- GQA (Visual Question Answering 데이터)
- Flickr30k (이미지-캡션 쌍)

**용도:**
- Region-text grounding 학습
- 자연어 설명과 이미지 영역 매칭
- Open-vocabulary 능력 강화

### 3.4 RAM++ 태그셋 (4585개)

**개요:**
- Recognize Anything Model Plus
- 이미지 태깅을 위한 대규모 레이블 시스템
- YOLOE의 Prompt-free 모드에서 사용

**구축 과정:**
```
1. 공개 데이터셋 태그 수집 (분류, 검출, 세그멘테이션)
2. 상용 태깅 서비스 태그 수집 (Google, Microsoft, Apple)
3. 텍스트에서 공통 태그 추출
4. 동의어 그룹화 (6449개 → 4585개 고유 ID)
5. WordNet, 번역, 수동 검토로 정제
```

**태그 예시:**
```
사람 관련: person, man, woman, child, worker, ...
안전 관련: helmet, hard hat, safety vest, ...
재난 관련: fire, flame, smoke, explosion, ...
행동 관련: standing, sitting, lying, falling, ...
```

---

## 4. Fine-tuning 전략

### 4.1 Trainer 클래스 종류

Ultralytics YOLOE는 여러 Trainer 클래스를 제공합니다:

| Trainer | 설명 | 용도 |
|---------|------|------|
| `YOLOETrainer` | 기본 YOLOE 트레이너 | 처음부터 학습 |
| `YOLOEPETrainer` | Linear Probing | 마지막 레이어만 학습 |
| `YOLOEPESegTrainer` | 세그멘테이션 Linear Probing | 세그멘테이션 fine-tuning |
| `YOLOETrainerFromScratch` | 처음부터 학습 | 텍스트 임베딩 포함 |
| `YOLOEPEFreeTrainer` | Prompt-free Linear Probing | LRPC 모듈 fine-tuning |
| `YOLOEVPTrainer` | Visual Prompt 학습 | SAVPE 인코더 학습 |

### 4.2 Linear Probing (YOLOEPETrainer)

**원리:**
- 대부분의 레이어 동결 (freeze)
- Classification head의 마지막 convolution만 학습
- 사전학습된 특징 추출기 유지

**장점:**
- 매우 빠른 학습 (10 epochs 충분)
- 적은 데이터로도 효과적
- Overfitting 위험 낮음
- 80%+ 성능 달성 가능

**단점:**
- 최고 성능 도달 어려움
- 도메인 갭이 큰 경우 한계

**코드:**
```python
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer

model = YOLOE("yoloe-26l-seg.pt")
results = model.train(
    data="custom.yaml",
    epochs=10,
    batch=32,
    imgsz=640,
    trainer=YOLOEPETrainer,
    # Linear Probing 최적 설정
    lr0=0.002,          # 높은 학습률 (마지막 레이어만)
    weight_decay=0.025,
    warmup_epochs=1,
)
```

### 4.3 Full Tuning

**원리:**
- 모든 파라미터 학습 가능
- 전체 네트워크 fine-tuning

**장점:**
- 최고 성능 달성 가능
- 도메인 적응력 높음

**단점:**
- 더 많은 학습 시간/데이터 필요
- Overfitting 주의
- Catastrophic forgetting 가능

**코드:**
```python
model = YOLOE("yoloe-26l-seg.pt")
results = model.train(
    data="custom.yaml",
    epochs=80,
    batch=16,
    imgsz=640,
    trainer=YOLOEPETrainer,  # 동일 trainer, epochs만 증가
    # Full Tuning 최적 설정
    lr0=0.001,          # 낮은 학습률 (전체 가중치 보존)
    weight_decay=0.0005,
    warmup_epochs=3,
)
```

### 4.4 권장 학습 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Zero-shot 테스트                                        │
│ - 사전학습 모델로 바로 테스트                                    │
│ - 클래스 이름만 변경해서 성능 확인                               │
│ - 기준선(baseline) 설정                                         │
├─────────────────────────────────────────────────────────────────┤
│ Step 2: Linear Probing (10 epochs)                              │
│ - 빠른 실험으로 데이터/설정 검증                                 │
│ - 클래스 이름, 데이터 품질 효과 확인                            │
│ - 여러 설정 빠르게 비교                                         │
├─────────────────────────────────────────────────────────────────┤
│ Step 3: Full Tuning (80 epochs)                                 │
│ - Linear Probing 최적 설정으로 진행                             │
│ - 최종 성능 최적화                                              │
│ - 필요시 이미지 해상도 증가 (1280)                              │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 학습 설정 상세

#### 하이퍼파라미터 권장값

| 설정 | Linear Probing | Full Tuning | 설명 |
|------|----------------|-------------|------|
| epochs | 10 | 80 | 소형 모델은 160 |
| batch | 32+ | 16-32 | GPU 메모리 최대 |
| imgsz | 640 | 640-1280 | 작은 객체 → 1280 |
| lr0 | 0.002 | 0.001 | 초기 학습률 |
| lrf | 0.1 | 0.01 | 최종 학습률 비율 |
| weight_decay | 0.025 | 0.0005 | L2 정규화 |
| warmup_epochs | 1 | 3 | 학습률 웜업 |
| patience | 5 | 20 | Early stopping |
| optimizer | AdamW | AdamW | 최적화 알고리즘 |

#### 데이터 증강 설정

```python
# Linear Probing (약한 증강)
augmentation_light = {
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
}

# Full Tuning (강한 증강)
augmentation_strong = {
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
}
```

---

## 5. 클래스 네이밍 전략

### 5.1 CLIP 텍스트 인코더의 이해

**CLIP (Contrastive Language-Image Pre-training):**
- OpenAI가 개발한 vision-language 모델
- 4억 개의 이미지-텍스트 쌍으로 학습
- 텍스트와 이미지를 동일한 임베딩 공간에 매핑

**MobileCLIP-B(LT):**
- YOLOE에서 사용하는 경량화 CLIP
- 모바일/엣지 디바이스에 적합
- 원본 CLIP 대비 빠른 추론

**텍스트 인코딩 과정:**
```
입력: "person wearing helmet"
    ↓
토큰화: ["person", "wearing", "helmet"]
    ↓
Transformer 인코더
    ↓
텍스트 임베딩 (512차원 벡터)
    ↓
이미지 임베딩과 코사인 유사도 계산
```

### 5.2 효과적인 클래스 이름 작성 원칙

#### 원칙 1: 자연어 사용

```python
# Bad - 프로그래머 스타일
classes = ["Helmet_OFF", "Helmet_ON", "NO_HELMET"]

# Good - 자연어 설명
classes = ["person without helmet", "person wearing helmet"]
```

**이유:**
- CLIP은 자연어 캡션으로 학습됨
- 언더스코어, 대문자는 학습 데이터에 드묾
- 자연어가 더 풍부한 의미 전달

#### 원칙 2: 문맥 포함

```python
# Bad - 객체만
classes = ["helmet", "no helmet"]

# Good - 주체 + 상태
classes = ["person without helmet", "person wearing helmet"]

# Better - 더 구체적인 문맥
classes = ["construction worker without safety helmet",
           "construction worker wearing yellow hard hat"]
```

#### 원칙 3: CLIP 프롬프트 템플릿 활용

CLIP은 학습 시 다양한 프롬프트 템플릿 사용:
```
"a photo of a {class}"
"a photograph of a {class}"
"an image of a {class}"
"a picture of a {class}"
...
```

**적용:**
```python
# 내부적으로 YOLOE는 클래스명을 그대로 사용
# 필요시 명시적 프롬프트 추가 가능
classes = ["a photo of fire", "a photo of smoke"]
```

#### 원칙 4: 사전학습 데이터 클래스와 유사하게

Objects365/LVIS의 클래스명 스타일 참고:
```
Objects365: "Person", "Helmet", "Fire Hydrant"
LVIS: "person", "hard_hat", "fire_extinguisher"
```

**권장:**
```python
# Objects365 스타일에 맞춤
classes = ["person", "helmet", "fire"]

# 또는 더 설명적으로
classes = ["person", "person wearing helmet", "fire flames"]
```

### 5.3 Null Class 전략

**개념:**
- 오탐을 유발하는 유사 객체를 별도 클래스로 추가
- 모델이 구분하도록 학습
- 추론 시 해당 클래스 무시

**예시:**
```yaml
# fire 오탐 방지
names:
  - 'fire'              # 타겟
  - 'orange light'      # null class - 유사하지만 다른 객체
  - 'sunset'            # null class
  - 'candle flame'      # null class (필요시 분리)

# smoke 오탐 방지
names:
  - 'smoke'             # 타겟
  - 'fog'               # null class
  - 'steam'             # null class
  - 'cloud'             # null class
```

**추론 시 필터링:**
```python
# null class 제외
target_classes = ['fire', 'smoke']
null_classes = ['orange light', 'sunset', 'fog', 'steam']

for det in detections:
    if det['class_name'] not in null_classes:
        # 실제 검출 결과로 사용
```

### 5.4 동의어 및 변형어 활용

CLIP은 동의어를 이해합니다:

```python
# 모두 유사한 임베딩 생성
"helmet" ≈ "hard hat" ≈ "safety helmet" ≈ "construction helmet"
"fire" ≈ "flames" ≈ "blaze" ≈ "burning"
"smoke" ≈ "fumes" ≈ "smog"
```

**전략:**
```python
# 가장 일반적인 용어 사용
classes = ["helmet"]  # "hard hat"보다 범용적

# 또는 구체적 용어 (도메인 특화)
classes = ["construction hard hat"]  # 건설 현장 특화
```

### 5.5 우리 프로젝트 클래스 최적화

**현재 → 권장:**

| 현재 | 문제점 | 권장 | 이유 |
|------|--------|------|------|
| `Helmet_OFF` | 언더스코어, 축약어 | `person without helmet` | 자연어, 주체 명시 |
| `Helmet_ON` | 언더스코어, 축약어 | `person wearing helmet` | 자연어, 행동 명시 |
| `fire` | OK | `fire` | Objects365에 없지만 CLIP 이해 |
| `smoke` | OK | `smoke` | CLIP 이해 |
| `fall` | 동사, 모호함 | `fallen person` | 상태 명시 |
| `person` | OK | `person` | Objects365와 일치 |

**최종 권장 구성:**
```yaml
names:
  - 'person without helmet'    # 0
  - 'person wearing helmet'    # 1
  - 'fire'                     # 2
  - 'smoke'                    # 3
  - 'fallen person'            # 4
  - 'person'                   # 5
```

### 5.6 실험적 대안

**더 상세한 설명:**
```yaml
names:
  - 'construction worker not wearing safety helmet'
  - 'construction worker wearing yellow hard hat'
  - 'fire flames burning'
  - 'smoke cloud rising'
  - 'person lying on the ground'
  - 'standing person'
```

**색상/크기 포함:**
```yaml
names:
  - 'person without yellow helmet'
  - 'person wearing yellow safety helmet'
  - 'small fire'
  - 'large fire'
  - 'white smoke'
  - 'black smoke'
```

---

## 6. 정확도 향상 방법론

### 6.1 데이터 품질 개선

#### 노이즈 이미지 제거

```python
# 제거해야 할 이미지 패턴
bad_patterns = [
    'mask', 'corona', 'covid',     # COVID 마스크 이미지
    'selfie', 'portrait',          # 셀카, 인물 사진
    'cartoon', 'drawing', 'anime', # 만화, 그림
    'logo', 'icon',                # 로고, 아이콘
]

def filter_bad_images(image_path):
    filename = image_path.stem.lower()
    return not any(p in filename for p in bad_patterns)
```

#### 작은 객체 필터링

```python
# 너무 작은 바운딩박스 제거
MIN_BBOX_AREA = 0.005  # 이미지 면적의 0.5%

def filter_small_objects(label_path):
    with open(label_path) as f:
        lines = []
        for line in f:
            parts = line.strip().split()
            w, h = float(parts[3]), float(parts[4])
            if w * h >= MIN_BBOX_AREA:
                lines.append(line)
    return lines
```

#### 클래스 불균형 해소

```python
# 클래스별 샘플 수 확인
class_counts = {
    'person without helmet': 15000,
    'person wearing helmet': 45000,  # 3배 많음
    'fire': 8000,
    'smoke': 6000,
    'fallen person': 2000,           # 매우 적음
    'person': 50000,
}

# 해결 방법:
# 1. Oversampling: 적은 클래스 복제
# 2. Undersampling: 많은 클래스 일부 제거
# 3. Class weights: 손실 함수에 가중치
# 4. Focal loss: 어려운 샘플에 집중
```

### 6.2 데이터 증강 강화

#### 기본 증강 (Ultralytics 내장)

```python
augmentation = {
    # 색상 변환
    "hsv_h": 0.015,     # Hue ±1.5%
    "hsv_s": 0.7,       # Saturation ±70%
    "hsv_v": 0.4,       # Value ±40%

    # 기하학적 변환
    "degrees": 10.0,    # 회전 ±10도
    "translate": 0.1,   # 이동 ±10%
    "scale": 0.5,       # 스케일 0.5~1.5
    "shear": 2.0,       # 전단 ±2도
    "perspective": 0.0, # 원근 변환

    # 뒤집기
    "flipud": 0.0,      # 상하 뒤집기 (보통 비활성화)
    "fliplr": 0.5,      # 좌우 뒤집기 50%

    # 고급 증강
    "mosaic": 1.0,      # 4개 이미지 합성
    "mixup": 0.1,       # 2개 이미지 블렌딩
    "copy_paste": 0.0,  # 객체 복사-붙여넣기
}
```

#### 도메인 특화 증강

```python
# Albumentations 사용 예시
import albumentations as A

transform = A.Compose([
    # 조명 변화 (실내/실외)
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.3),

    # 날씨 효과
    A.RandomRain(p=0.1),       # 비
    A.RandomFog(p=0.1),        # 안개
    A.RandomSunFlare(p=0.1),   # 햇빛 반사

    # 카메라 효과
    A.GaussNoise(p=0.2),       # 노이즈
    A.MotionBlur(p=0.1),       # 움직임 블러
    A.ImageCompression(p=0.2), # JPEG 압축

    # 화재/연기 특화
    A.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.5,  # 화재 색상 변화
        hue=0.1,
        p=0.5
    ),
])
```

### 6.3 학습 설정 최적화

#### 이미지 해상도

| 해상도 | 장점 | 단점 | 권장 상황 |
|--------|------|------|-----------|
| 640 | 빠른 학습/추론 | 작은 객체 놓침 | 기본, 큰 객체 |
| 960 | 균형 | - | 중간 크기 객체 |
| 1280 | 작은 객체 검출↑ | 메모리/시간↑ | 작은 객체 많음 |

```bash
# 작은 객체가 많은 경우
python train_unified.py --imgsz 1280 --batch 8
```

#### 배치 크기

```python
# GPU 메모리에 따른 권장 배치 크기
# RTX 3090 (24GB): batch=32 (640px), batch=8 (1280px)
# RTX 4090 (24GB): batch=48 (640px), batch=12 (1280px)
# A100 (40GB): batch=64 (640px), batch=16 (1280px)

# 작은 배치 문제: BatchNorm 통계 불안정
# 해결: SyncBatchNorm (멀티 GPU) 또는 배치 크기 증가
```

#### Learning Rate 스케줄

```python
# Cosine Annealing (기본)
# lr = lr0 * (1 + cos(π * epoch / epochs)) / 2 * (1 - lrf) + lrf

# 설정
train_config = {
    "lr0": 0.001,      # 초기 학습률
    "lrf": 0.01,       # 최종 학습률 = lr0 * lrf
    "warmup_epochs": 3, # 웜업 기간
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
}
```

### 6.4 YOLOE 특화 방법

#### Visual Prompt 활용

```python
from ultralytics import YOLOE
import cv2
import glob

# 클래스별 참조 이미지 로드
def load_reference_images(class_name, image_dir, n=16):
    """클래스당 N개의 대표 이미지 로드"""
    images = []
    for path in glob.glob(f"{image_dir}/{class_name}/*.jpg")[:n]:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images

# Visual prompt embedding 생성
model = YOLOE("yoloe-26l-seg.pt")

reference_data = {
    "fire": load_reference_images("fire", "references"),
    "smoke": load_reference_images("smoke", "references"),
    "fallen person": load_reference_images("fall", "references"),
}

# Visual embedding 생성 및 적용
visual_pe = model.get_visual_pe(reference_data)
model.set_classes(list(reference_data.keys()), visual_pe)

# 추론
results = model.predict("test_image.jpg")
```

#### 클래스별 Confidence Threshold

```python
# 클래스별 최적 threshold (검증 데이터로 튜닝)
class_thresholds = {
    'person without helmet': 0.45,
    'person wearing helmet': 0.50,
    'fire': 0.30,               # 사전학습 데이터 부족 → 낮게
    'smoke': 0.25,              # 사전학습 데이터 부족 → 낮게
    'fallen person': 0.35,
    'person': 0.50,
}

def filter_by_class_threshold(detections, thresholds):
    """클래스별 threshold 적용"""
    filtered = []
    for det in detections:
        cls_name = det['class_name']
        threshold = thresholds.get(cls_name, 0.5)
        if det['confidence'] >= threshold:
            filtered.append(det)
    return filtered
```

#### Threshold 자동 튜닝

```python
from sklearn.metrics import precision_recall_curve

def find_optimal_threshold(y_true, y_scores, target_precision=0.9):
    """목표 precision에서의 최적 threshold 찾기"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # target_precision 이상인 threshold 중 가장 낮은 것
    valid_idx = precision >= target_precision
    if valid_idx.any():
        return thresholds[valid_idx][0]
    return 0.5  # 기본값
```

### 6.5 추론 최적화

#### Test Time Augmentation (TTA)

```python
# TTA 활성화 - 다중 스케일 추론
results = model.predict(
    "image.jpg",
    augment=True,  # TTA 활성화
)

# TTA 동작:
# 1. 원본 이미지 추론
# 2. 좌우 뒤집힌 이미지 추론
# 3. 다른 스케일로 추론
# 4. 결과 앙상블 (NMS)
```

#### 앙상블

```python
from ultralytics import YOLOE
import numpy as np

# 여러 모델 로드
models = [
    YOLOE("yoloe-26l-best.pt"),
    YOLOE("yoloe-26m-best.pt"),
    YOLOE("yoloe-26l-best-v2.pt"),
]

def ensemble_predict(models, image, conf=0.3, iou=0.5):
    """여러 모델의 결과 앙상블"""
    all_boxes = []
    all_scores = []
    all_classes = []

    for model in models:
        results = model.predict(image, conf=conf)[0]
        if results.boxes is not None:
            all_boxes.extend(results.boxes.xyxy.cpu().numpy())
            all_scores.extend(results.boxes.conf.cpu().numpy())
            all_classes.extend(results.boxes.cls.cpu().numpy())

    # NMS로 중복 제거
    if all_boxes:
        # torchvision.ops.nms 또는 직접 구현
        keep = nms(all_boxes, all_scores, iou)
        return [all_boxes[i] for i in keep]
    return []
```

### 6.6 정확도 향상 체크리스트

```markdown
## 데이터
- [ ] 노이즈 이미지 제거 (마스크, 셀카, 만화 등)
- [ ] 작은 객체 필터링 (< 0.5% 면적)
- [ ] 라벨링 오류 검수
- [ ] 클래스 불균형 확인 및 조정
- [ ] 충분한 데이터 양 확보 (클래스당 1000+ 권장)

## 클래스 네이밍
- [ ] 자연어 스타일로 변경
- [ ] 사전학습 데이터셋 클래스 참고
- [ ] Null class 추가 고려

## 학습 설정
- [ ] 적절한 이미지 해상도 선택
- [ ] 배치 크기 최대화
- [ ] Linear Probing → Full Tuning 순서
- [ ] Early stopping patience 설정

## 추론 최적화
- [ ] 클래스별 confidence threshold 튜닝
- [ ] TTA 적용 고려
- [ ] 앙상블 고려 (정확도 최우선 시)

## 검증
- [ ] 검증 데이터로 mAP 측정
- [ ] 클래스별 AP 확인
- [ ] 실제 환경 테스트
```

---

## 7. Resume 학습 문제와 해결

### 7.1 문제 원인

YOLOE에서 `resume=True` 사용 시 발생하는 에러:
```
ValueError: loaded state dict contains a parameter group
that doesn't match the size of optimizer's group
```

**원인:**
1. YOLOE는 일반 YOLO와 다른 구조 (텍스트 임베딩 레이어 포함)
2. Optimizer state에 추가 파라미터 그룹 존재
3. Resume 시 optimizer 복원 과정에서 불일치

### 7.2 Resume vs 새 학습

| 방식 | Optimizer | Epoch 카운터 | LR 스케줄 |
|------|-----------|--------------|-----------|
| Resume | 복원 시도 | 이어서 | 이어서 |
| 새 학습 | 새로 초기화 | 0부터 | 처음부터 |

### 7.3 해결 방법

#### 방법 1: 가중치만 로드 (권장)

```python
from ultralytics import YOLOE

# 기존 best.pt에서 가중치만 로드
model = YOLOE("runs/unified/exp/weights/best.pt")

# 새로운 학습 시작 (optimizer 새로 초기화)
model.train(
    data="data.yaml",
    epochs=50,  # 추가 학습할 epoch
    # resume=False (기본값)
)
```

#### 방법 2: strip_optimizer 사용

```python
from ultralytics.utils.torch_utils import strip_optimizer

# checkpoint에서 optimizer 상태 제거
strip_optimizer("runs/unified/exp/weights/last.pt")

# 이후 일반 학습으로 사용
model = YOLOE("runs/unified/exp/weights/last.pt")
model.train(data="data.yaml", epochs=50)
```

#### 방법 3: 처음부터 충분한 설정

```python
# Resume 대신 처음부터 충분한 epochs + patience
model.train(
    data="data.yaml",
    epochs=100,      # 충분히 크게
    patience=30,     # Early stopping
)
```

### 7.4 권장 워크플로우

```
1. 처음 학습 시 충분한 epochs 설정 (100+)
2. patience로 Early stopping 활성화 (20-30)
3. 중단 시 best.pt에서 새 학습 시작
4. epoch 수는 추가로 학습할 만큼만 설정
```

---

## 8. 실전 사용법

### 8.1 환경 설정

```bash
# 필수 패키지
pip install ultralytics>=8.0.0
pip install torch>=2.0.0

# 선택 패키지
pip install albumentations  # 추가 증강
pip install wandb           # 실험 추적
```

### 8.2 데이터셋 구조

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**data.yaml 예시:**
```yaml
train: ./train/images
val: ./valid/images
test: ./test/images

nc: 6
names:
  - 'person without helmet'
  - 'person wearing helmet'
  - 'fire'
  - 'smoke'
  - 'fallen person'
  - 'person'
```

### 8.3 학습 명령어

```bash
# Linear Probing (빠른 검증)
python train_unified.py --mode linear --model yoloe-26l

# Full Tuning (성능 최적화)
python train_unified.py --mode full --model yoloe-26l

# 고해상도 학습
python train_unified.py --mode full --imgsz 1280 --batch 8

# 기존 가중치에서 계속
python train_unified.py --mode full --weights runs/unified/exp/weights/best.pt
```

### 8.4 추론

```python
from ultralytics import YOLOE

# 모델 로드
model = YOLOE("runs/unified/best.pt")

# 이미지 추론
results = model.predict("image.jpg", conf=0.3)

# 결과 처리
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f"Class: {r.names[cls]}, Conf: {conf:.2f}, Box: {xyxy}")

# 결과 시각화
results[0].show()
results[0].save("result.jpg")
```

### 8.5 RTSP 스트림 추론

```python
from ultralytics import YOLOE
import cv2

model = YOLOE("runs/unified/best.pt")

# RTSP 스트림 열기
cap = cv2.VideoCapture("rtsp://user:pass@ip:port/stream")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 추론
    results = model.predict(frame, conf=0.3, verbose=False)

    # 시각화
    annotated = results[0].plot()
    cv2.imshow("Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 9. 트러블슈팅

### 9.1 CUDA Out of Memory

```python
# 해결 방법
# 1. 배치 크기 줄이기
--batch 16  # → --batch 8

# 2. 이미지 크기 줄이기
--imgsz 1280  # → --imgsz 640

# 3. Mixed precision 활성화 (기본 활성화)
--amp True

# 4. Gradient checkpointing
# (ultralytics에서 자동 관리)
```

### 9.2 낮은 mAP

```markdown
체크리스트:
1. 데이터 품질 확인 (라벨링 오류?)
2. 클래스 불균형 확인
3. 클래스 이름 자연어로 변경했는지
4. 충분한 데이터 양인지 (클래스당 500+ 권장)
5. 이미지 해상도 적절한지 (작은 객체 → 1280)
6. 학습률 적절한지
7. 충분한 epoch 학습했는지
```

### 9.3 특정 클래스 성능 저하

```python
# 클래스별 AP 확인
results = model.val(data="data.yaml")
print(results.box.ap_class_index)  # 클래스별 AP

# 해결:
# 1. 해당 클래스 데이터 추가
# 2. 해당 클래스 oversampling
# 3. 클래스 이름 변경 (더 설명적으로)
# 4. Visual prompt 사용
```

### 9.4 높은 오탐률 (False Positive)

```python
# 해결 방법:
# 1. Confidence threshold 높이기
model.predict(image, conf=0.5)  # 0.3 → 0.5

# 2. Null class 추가
# data.yaml에 유사 객체 클래스 추가

# 3. Hard negative mining
# 오탐 이미지를 negative sample로 추가
```

### 9.5 느린 추론 속도

```python
# 해결 방법:
# 1. 작은 모델 사용
model = YOLOE("yoloe-26s-seg.pt")  # L → S

# 2. 이미지 크기 줄이기
model.predict(image, imgsz=416)

# 3. Half precision
model.predict(image, half=True)

# 4. TensorRT 변환
model.export(format="engine")
```

---

## 10. 참고 자료

### 10.1 공식 문서

- [YOLOE 논문 (ICCV 2025)](https://arxiv.org/html/2503.07465v1)
- [YOLOE GitHub (THU-MIG)](https://github.com/THU-MIG/yoloe)
- [Ultralytics YOLOE 문서](https://docs.ultralytics.com/models/yoloe/)
- [YOLO-World 논문 (CVPR 2024)](https://arxiv.org/abs/2401.17270)
- [YOLO-World GitHub](https://github.com/AILab-CVC/YOLO-World)

### 10.2 데이터셋

- [Objects365](https://www.objects365.org/)
- [LVIS Dataset](https://www.lvisdataset.org/)
- [COCO Dataset](https://cocodataset.org/)

### 10.3 관련 기술

- [CLIP (OpenAI)](https://openai.com/research/clip)
- [MobileCLIP](https://github.com/apple/ml-mobileclip)
- [RAM++ (Recognize Anything)](https://github.com/xinyu1205/recognize-anything)

### 10.4 튜토리얼 및 가이드

- [YOLO-World Fine-tuning Guide](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/finetuning.md)
- [YOLO World Prompting Tips (Roboflow)](https://blog.roboflow.com/yolo-world-prompting-tips/)
- [Ultralytics Training Guide](https://docs.ultralytics.com/modes/train/)

### 10.5 커뮤니티

- [Ultralytics Discord](https://discord.gg/ultralytics)
- [Ultralytics GitHub Discussions](https://github.com/ultralytics/ultralytics/discussions)

---

> 문서 버전: 1.0
> 최종 수정: 2026-01-21
> 작성: Claude Code Assistant
