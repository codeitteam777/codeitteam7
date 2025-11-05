# 💊 [AI] 헬스잇(Health Eat) 알약 검출 프로젝트 (내부 실험)

헬스케어 스타트업 '헬스잇(Health Eat)'의 AI 엔지니어링 팀으로서, 사용자가 촬영한 알약 사진에서 최대 4개의 알약 이름과 위치를 검출하는 모델을 개발합니다.

## 🚀 최소 비용, 최고 안정성을 위한 알약 객체 검출 모델 최적화
| Stat | Value | | Stat | Value |
|------|-------|---|------|-------|
| **Final Test mAP@50-95** | **0.9808** | | **Precision** | **0.9862** |
| **Recall** | **0.9795** | | **Model** | **YOLOv8n (Nano)** |
| **Optimizer** | **AdamW** | | | |

## 1. 🛠️ 개발 및 분석 스택 (Development Stack)

| Category | Technology | Usage | Badge |
|----------|-----------|-------|-------|
| **Core Model** | YOLOv8 (Ultralytics) | 알약 객체 탐지 모델 베이스라인 | ![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=flat-square&logo=YOLO&logoColor=black) |
| **Deep Learning** | PyTorch | 모델 학습 및 추론 환경 | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) |
| **Language** | Python | 전체 데이터 파이프라인 및 모델링 | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Data Tool** | Roboflow | 의사 레이블링(Pseudo Labeling) 및 데이터 전처리 | ![Roboflow](https://img.shields.io/badge/Roboflow-6706CE?style=flat-square&logo=roboflow&logoColor=white) |
| **Analysis** | Scikit-learn, OpenCV | t-SNE 특징 공간 분석 및 이미지 처리 | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) |



## 2. 👥 우리 팀 (Team)

| 이름 | 역할 |
| :--- | :--- |
| **이승완(팀장, PM)** | EDA, 초기 베이스라인 코드, 의사 레이블링, 모델링 (Optimizer) |
| **이경식** | 모델링 (Model Size / Version) |
| **오현민** | 의사 레이블링, 모델링 (ImgSize / LR) |
| **최준영** | 의사 레이블링, 모델링 (Augmentation) |
| **박병호** | EDA, 데이터 분석 |

## 3. 🎯 프로젝트 개요

**초기 문제**: yolov8s로 Kaggle 리더보드 점수 0.986을 달성했으나, Test 셋과 Train 셋이 거의 동일하여 mAP 지표가 변별력을 상실함을 확인.

**신규 목표**: mAP 0.99 달성을 기본값으로 간주. **"가장 낮은 Validation Loss(최고의 안정성)"**와 **"최소 학습 비용(Cost)"**을 확보하는 최적의 모델을 탐색하는 것으로 목표를 전환.

## 4. 📂 데이터 파이프라인

**EDA 및 정제**: 원본(1489개)에서 라벨 불일치 [오류셋 840개] 식별 및 이상치 17개 제거.

**의사 레이블링**: [오류셋 840개]의 라벨을 Roboflow로 복원.

**최종 데이터셋**: [클린셋 632개] + [복원셋 840개] = 총 1,472개의 통합 데이터셋 구축.

**분할**: Train (70%) / Val (15%) / Test (15%)로 계층 분할하여 내부 실험 진행.

## 5. 🔍 핵심 발견 (Key Findings)

**3대 편향 식별**: EDA 결과, 데이터셋에 '환경' (동일 조명/배경), '공간' (좌측 하단 밀집), '특징' 편향이 존재함을 확인.

**mAP 0.99의 진실**: t-SNE(실루엣 -0.49) 분석 결과, '색상/형태' 등 일반 특징으로는 클래스 구분이 불가능했음. YOLO가 mAP 0.99를 달성한 이유는 '일반 특징'이 아닌, 알약 표면의 **'텍스트 각인(Imprint)'**이라는 '미세 특징'에 과적합되었기 때문임을 규명.

## 6. 📊 프로젝트 구조

```
📦 project-root
├── 📂 docs/                       # 최종보고서 파일
├── 📂 config/                     # yaml 파일
├── 📂 img/                        # img 파일
├── 📂 legacy/                     # 이전 파일
├── 📂 data/                       # (개별 private 데이터)
│   ├── 📂 raw/                    # 원본 데이터
│   ├── 📂 processed/              # 전처리된 데이터
├── 📂 notebooks/
│   ├── 📓 01_EDA.ipynb           # 탐색적 데이터 분석
│   ├── 📓 02_Model_Baseline.ipynb# 기본 베이스라인 분석
├── 📂 src/                       # 실행소스 파일
├── 📂 team_src/                  # 팀원 개별 파일
│   ├── 📓 best_model_test.ipynb  # 최고 Score 파일
└── 📄 README.md
```


## 7. 📄 상세 보고서

상세한 보고서는 별도 문서를 참조.
## 보고서 다운로드
[보고서 PDF 다운로드](docs/Codeitteam7_AI_헬스잇(Health_Eat)_알약_검출_프로젝트_내부_실험_보고서.pdf)


## 개인 협업일지

이승완 : https://www.notion.so/292bfba1db0b80b88c23f3f789db0d73?source=copy_link

오현민 : https://www.notion.so/7-2a1e0d876d7580c085aae4b84acc4761?source=copy_link

이경식 : https://www.notion.so/2a23a594a4d080bd9543ce1842bf85c5?source=copy_link

박병호 : https://www.notion.so/2a25df876c2080499878e5c12d5d1e76?source=copy_link

최준영 :
