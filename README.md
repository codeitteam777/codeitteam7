# [AI] 헬스잇(Health Eat) 알약 검출 프로젝트 (내부 실험)

> 헬스케어 스타트업 '헬스잇(Health Eat)'의 AI 엔지니어링 팀으로서, 사용자가 촬영한 알약 사진에서 최대 4개의 알약 이름과 위치를 검출하는 모델을 개발합니다.

## 우리 팀 (Team)

| 이름 | 역할 | Github |
| :--- | :--- | :--- |
| **이승완** | **팀장 (PM)**, EDA, 초기 베이스라인 코드, 의사 레이블링, 모델링 (Optimizer) | |
| **이경식** | 모델링 (Model Size / Version) | |
| **오현민** | 모델링 (ImgSize / LR) | |
| **최준영** | 모델링 (Augmentation) | |
| **박병호** | EDA, 데이터 분석 | |

## 프로젝트 목표

* **1차 목표:** 알약 검출 모델(`mAP@[0.75:0.95]`) 개발.
* **현상 진단:** Baseline(`yolov8n`) 모델이 `Val` 셋(15%)에 대해 `mAP@75` **0.9946**을 기록.
* **핵심 문제:** `Val` 셋이 `Train` 셋과 통계적으로 동일하여(배경, 조명 편향), mAP 지표가 변별력을 상실함.
* **신규 목표:** `mAP 0.99` 달성을 기본값으로 간주. **"mAP 0.99를 달성하는 최소 비용(Cost)을 탐색"**하고, 동시에 **"가장 낮은 Validation Loss(최고의 안정성)"**를 확보하는 것을 핵심 목표로 설정.

## 프로젝트 환경

* **데이터셋:** 원본(632개) + 의사 라벨(840개) = **총 1,472개**의 통합 데이터셋.
* **데이터 분할:** `CombinedDataset` 폴더에 `Train (70%) / Val (15%) / Test (15%)`로 계층 분할하여 내부 실험 진행.
* **작업 방식:** 각 팀원은 개별 Jupyter Notebook (`.ipynb`) 파일로 실험을 수행하고 결과를 공유.

## 1. 데이터 파이프라인 (1472 Dataset)

### Phase 1: EDA 및 정제
* 원본(1489개) 분석 결과, 라벨링 개수가 파일명과 일치하는 **[클린셋 632개]**와 불일치하는 **[오류셋 840개]**를 식별.
* (IoU, OOB 등 17개 이상치 파일은 정제 과정에서 제거)

### Phase 2: 의사 레이블링 (Pseudo-Labeling)
* '오류셋(840개)'의 라벨을 Roboflow를 활용해 복원.

### Phase 3: 최종 데이터셋 구축
* **`Clean Dataset (632개)`** + **`Pseudo-Label Dataset (840개)`** = **총 1,472개**의 이미지/라벨을 `CombinedDataset` 폴더에 YOLO 포맷으로 통합 및 `7:1.5:1.5` 분할.

## 2. 주요 실험 결과 (Key Findings)

### Finding 1: EDA - "mAP 0.99는 당연한 결과" (담당: 이승완, 박병호)
* **데이터 동질성:** 모든 이미지가 `976x1280` 크기, `연회색 배경`, `주백색 조명`으로 완벽히 통제됨.
* **위치/크기 편향:** BBox 중심점이 이미지 중앙에 압도적으로 몰려있으며, 크기 또한 매우 유사함.
* **결론:** mAP 0.99 달성은 쉽다. **모델의 '안정성(Val Loss)'**이 진짜 승부처.

### Finding 2: Optimizer 실험 - "AdamW가 가장 안정적" (담당: 이승완)
* **목표:** mAP 0.99 달성을 전제로, Val Loss (Box, Cls, DFL)의 총합이 가장 낮은 옵티마이저를 선정.
* **설정:** yolov8n, epochs=70, imgsz=640, batch=16, augment=False, lr0=0.001
* **결과:** AdamW가 58 Epoch에서 가장 낮은 Validation Loss 합계를 기록.

| 옵티마이저 | mAP@75 | Val Box Loss (Best) | Val Cls Loss (Best) | Val DFL Loss (Best) | 총 시간 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AdamW** | **0.9925** | **0.1543 (at 58e)** | **0.1454 (at 58e)** | **0.7621 (at 58e)** | 1539s |
| **Nadam** | **0.9926** | 0.1716 | 0.1702 | 0.7661 | **1317s** |
| SGD | 0.9916 | 0.1718 | 0.2045 | 0.7736 | 1418s |
| Adam | 0.9896 | 0.1882 | 0.2107 | 0.7716 | 1122s |
| RMSprop | 0.6741 | 0.1886 | 0.8212 | 0.7874 | 1608s |

* **결론:** `Nadam`이 속도는 가장 빨랐으나, **`AdamW`**가 모든 Val Loss 지표에서 압도적으로 안정적인 성능(최저 Loss)을 보여줌. **`AdamW`를 Baseline 옵티마이저로 채택.**

### Finding 3: Model Cost 분석 - "YOLOv8n이 Cost-Effective" (담당: 이경식)
* **목표:** mAP@75 0.99 달성에 필요한 최소 시간(Cost) 비교 (안정성(Loss)과는 별개로 측정).
* **설정:** AdamW 기준, mAP 0.99 최초 도달 시간

| Model | Model Size | 0.99 최초 도달 Epoch | 0.99 최초 도달 시간 |
| :--- | :---: | :---: | :---: |
| yolo8 | n | 32 | 3분 44.5초 |
| yolo8 | s | 38 | 3분 52.9초 |
| ... | ... | ... | ... |
| yolo11 | n | 실패 (0.97 종료) | - |

* **결론:** 단순 mAP 0.99 달성 속도는 **`yolov8n`**이 가장 빠름. Finding 2의 안정성(Loss) 결과와 종합하여 `yolov8n`을 최종 Baseline 모델로 채택.

## 3. 실험 계획 (Phase 2 & 3)

### Phase 1: 기준 비용(Baseline Cost) 확립 (완료)
* **목표:** mAP@75 0.99 달성을 전제로, 가장 안정적인(Lowest Loss) 모델을 만들기 위한 기준 실험 환경 확립.
* **결과 (Finding 2, 3):**
  - `yolov8n`이 가장 효율적(Finding 3)
  - `AdamW`가 가장 안정적(Finding 2)
  - 안정화(Best Loss)에 58 Epochs가 소요되었으므로, 여유를 두어 **70 Epochs**를 실험 기준으로 설정.
* **팀 기준 Baseline:**
  - Model: `yolov8n`
  - Optimizer: `AdamW` (lr=0.001, augment=False)
  - Cost (Epochs): `70 Epochs` (실험 기준)
  - Loss (Target): `Val Box Loss 0.1543` (이보다 낮춰야 함)

### Phase 2: 비용 절감 및 Val Loss 최소화 (진행중)
* **공통 목표:** `yolov8n + AdamW + 70 Epochs` Baseline을 기준으로, 더 낮은 Val Loss를 달성하거나, 동일/유사 Loss를 더 적은 Epoch로 달성하는 설정 탐색.
* **이경식 (모델 크기):** (완료) `yolov8n`을 Baseline 모델로 채택.
* **오현민 (ImgSize / LR):** `ImgSize` (320, 640, 1280) 비교 / `AdamW` 기준 최적 LR 탐색.
* **최준영 (증강):** `Augment=True` (기본값) 설정이 `Augment=False` 대비 Val Loss 개선에 효과가 있는지 비교.
* **박병호 (EDA):** `train/val/test` 세트 간 클래스 분포, BBox 면적 분포 비교 시각화. (3개 세트가 사실상 동일함을 증명)

### Phase 3: 최종 검증 (Test 셋)
* **목표:** `7:1.5:1.5` 분할의 유의미성 검증.
* **실행:** Phase 2에서 **"Val Loss가 가장 낮았던"** 모델 (`best.pt`)을 최종 선정.
* 해당 모델로 `Test` 셋(15%)의 mAP를 단 1회 측정.
* **예상 결론:** "EDA 및 Test 셋 검증 결과, Train/Val/Test 셋이 통계적으로 동일함을 최종 확인. 7:1.5:1.5 분할은 일반화 검증에 유의미하지 않음."

## 4. 다음 단계 (Next Steps)

- [✓] **Baseline 확정 (PM):** `yolov8n`, `AdamW`, `70 Epochs`, `Val Box Loss 0.1543`
- [ ] **오현민, 최준영:** Phase 2 실험 시작. 70 Epoch Baseline 기준으로 ImgSize, Augmentation 실험 진행 및 결과 공유.
- [ ] **박병호:** `train/val/test` 3개 셋 분포 동일성 시각화 자료 공유.
- [ ] **전원:** Phase 3 실행 및 최종 결론 도출.
