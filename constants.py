import os

# 1. 경로 상수 (사용자 환경에 맞게 이 값만 수정합니다)
PROJECT_ROOT = "/Users/bellboi/code/codeitteam7"
# r"C:\Users\daboi\Desktop\ai05-level1-project"

# 데이터 폴더 경로 (PROJECT_ROOT 기준)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
TEMP_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "temp")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs") # outputs 경로를 절대 경로로 정의

# 특정 데이터 폴더
TRAIN_IMG_DIR = os.path.join(PROJECT_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(PROJECT_ROOT, "test_images")

# 모델 저장 및 제출 경로 (Exp 대신 Outputs 사용 권장)
EXP_BASE_DIR = os.path.join(PROJECT_ROOT, "Exp", "codeitteam7exp1") # 기존 Exp 유지
SUBMISSION_DIR = os.path.join(EXP_BASE_DIR, "submissions")

# IoU 및 OOB 분석 결과, 학습 데이터셋에서 제외할 오류 파일 목록
IOU_ERROR_FILES = [
    "K-003351-018147-020238_0_2_0_2_90_000_200.png",
    "K-003483-027733-030308-036637_0_2_0_2_90_000_200.png",
    "K-003351-020238-031863_0_2_0_2_70_000_200.png",
    "K-003351-029667-031863_0_2_0_2_70_000_200.png",
    "K-003483-019861-025367-029667_0_2_0_2_90_000_200.png",
    "K-002483-003743-012081-019552_0_2_0_2_90_000_200.png",
    "K-003483-019861-020238-031885_0_2_0_2_70_000_200.png",
    "K-003351-003832-029667_0_2_0_2_90_000_200.png",
    "K-001900-016548-019607-033009_0_2_0_2_70_000_200.png"
]

OOB_ERROR_FILES = [
    "K-003351-016262-018357_0_2_0_2_75_000_200.png",
    "K-003544-004543-012247-016551_0_2_0_2_70_000_200.png"
]

FILES_TO_EXCLUDE = IOU_ERROR_FILES + OOB_ERROR_FILES

# 모델 관련 상수
YOLO_VER = 'yolov8s.pt'
NUM_CLASSES = 73
IMAGE_SIZE = (976,1280)

# 데이터 분할 상수
TEST_SIZE = 0.2
RANDOM_STATE = 42
