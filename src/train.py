# @title train.py
import os
import re
import shutil
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# --- 프로젝트 모듈 Import ---
# PyCharm 환경에서 src 폴더가 Content Root로 설정되어 있다면 다음과 같이 import합니다.
# 실제 환경에 따라 import 경로는 조정될 수 있습니다.
# (예: from preprocessing.dataset import ...)
try:
    from src.preprocessing.dataset import get_cleaned_data_and_counts, create_yolo_dataset_structure
    from src.utils.data_split_utils import prepare_stratified_split
    # *참고: 실제 환경에서는 load_master_data 함수도 별도로 import 해야 합니다.
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("WARNING: Assuming helper functions are available for structure demonstration.")

# --------------------------------------------------------------------------------------
# 1. CONFIGURATION: 경로 및 하이퍼파라미터 정의
# --------------------------------------------------------------------------------------

# NOTE: 이 경로는 사용자님의 로컬 환경 경로를 정확히 반영해야 합니다.
PROJECT_ROOT = r"C:\Users\daboi\Desktop\ai05-level1-project"

config = {
    # --- 경로 설정 ---
    'project_root': PROJECT_ROOT,
    'data_dir': PROJECT_ROOT,
    'train_img_dir': os.path.join(PROJECT_ROOT, "train_images"),
    'test_img_dir': os.path.join(PROJECT_ROOT, "test_images"),
    'base_dir': os.path.join(PROJECT_ROOT, "Exp", "codeitteam7exp1"),  # 모델, 제출물 저장 기본 폴더
    'model_save_dir': os.path.join(PROJECT_ROOT, "Exp", "codeitteam7exp1", "models"),
    'submission_dir': os.path.join(PROJECT_ROOT, "Exp", "codeitteam7exp1", "submissions"),

    # --- 데이터 분할 설정 ---
    'test_size': 0.2,  # Val set 비율
    'random_state': 42,  # 재현성을 위한 시드

    # --- 학습 하이퍼파라미터 (YOLO) ---
    'model_name': 'yolov8s.pt',
    'epochs': 200,
    'imgsz': 640,
    'batch': 8,
    'device': 0,  # GPU 0번 사용
    'project_name': 'yolo_v1',
    'patience': 20,
    'lr0': 0.01,
    'optimizer': 'AdamW',
    'augment': True,
    'close_mosaic': 10
}

# 폴더 생성
os.makedirs(config['model_save_dir'], exist_ok=True)
os.makedirs(config['submission_dir'], exist_ok=True)

print("=" * 60)
print("경로 및 설정 완료")
print("=" * 60)

# --------------------------------------------------------------------------------------
# *가정: 데이터 로드 함수 (load_master_data)를 통해
#       master_data와 class_to_id 맵핑이 로드되었다고 가정하고 다음 단계를 진행합니다.
#       실제로는 여기에 master_data 로드 코드를 작성해야 합니다.
# --------------------------------------------------------------------------------------
# # 예시 데이터 구조 (실제 코드가 아닙니다!)
# master_data = load_master_data(os.path.join(config['data_dir'], "train.json"))
# class_to_id = {'class_a': 0, 'class_b': 1, ...}

# --------------------------------------------------------------------------------------
# 2. 데이터 전처리 파이프라인 호출
# --------------------------------------------------------------------------------------

try:
    # 1단계: 데이터 정제 및 클래스 카운팅 (dataset.py)
    # clean_master_data, class_count_lookup = get_cleaned_data_and_counts(master_data)

    # 2단계: Stratified Split을 위한 준비 (data_split_utils.py)
    # x_filenames, y_stratify_labels = prepare_stratified_split(clean_master_data, class_count_lookup)

    # 3단계: YOLO 데이터셋 구축 (dataset.py)
    # yaml_path = create_yolo_dataset_structure(
    #     clean_master_data,
    #     x_filenames,
    #     y_stratify_labels,
    #     class_to_id, # class_to_id는 data.yaml 생성에 필요
    #     config
    # )

    # *참고: 실제 학습을 위해 위 3단계를 주석 해제하고, 아래 yaml_path는 최종 경로로 대체하세요.
    yaml_path = os.path.join(config['data_dir'], "yolo_dataset", "data.yaml")

except NameError:
    # 모듈화 함수를 사용할 수 없는 경우, 임시로 경로만 지정합니다.
    print("\n[WARNING] 모듈화된 전처리 함수를 호출할 수 없습니다. YAML 경로만 지정합니다.")
    yaml_path = os.path.join(config['data_dir'], "yolo_dataset", "data.yaml")
    # 아래 추론 로직을 위해 임시 변수를 정의합니다.
    test_image_mapping = {}  # 이 부분은 추론 로직에서 다시 정의됩니다.
    class_to_id = {}  # 추론 시 category_id + 1을 위해 필요한 정보

# --------------------------------------------------------------------------------------
# 3. 모델 학습 (Train)
# --------------------------------------------------------------------------------------

print("\n학습 시작")
model = YOLO(config['model_name'])

results = model.train(
    data=yaml_path,
    epochs=config['epochs'],
    imgsz=config['imgsz'],
    batch=config['batch'],
    device=config['device'],
    project=config['model_save_dir'],
    name=config['project_name'],
    exist_ok=True,
    patience=config['patience'],
    lr0=config['lr0'],
    optimizer=config['optimizer'],
    augment=config['augment'],
    close_mosaic=config['close_mosaic']
)

best_pt = os.path.join(config['model_save_dir'], config['project_name'], 'weights', 'best.pt')
print(f"\n학습 완료 모델 저장: {best_pt}")

# --------------------------------------------------------------------------------------
# 4. 모델 검증 (Validation)
# --------------------------------------------------------------------------------------

print("\n모델 검증")
# 학습된 best 모델 로드
if os.path.exists(best_pt):
    model = YOLO(best_pt)
    metrics = model.val()
    print(f"mAP50     : {metrics.box.map50:.3f}")
    print(f"mAP50-95  : {metrics.box.map:.3f}")
else:
    print("WARNING: best.pt 파일 경로를 찾을 수 없어 검증을 건너뜁니다.")

# --------------------------------------------------------------------------------------
# 5. Test 추론 & 제출 파일 생성
# --------------------------------------------------------------------------------------

print("\nTest 추론 시작")
submission_data = []
annotation_id = 1

test_img_dir = config['test_img_dir']
test_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png')])

# 테스트 이미지 ID 매핑 생성 (원본 코드 유지)
test_image_mapping = {}
for img_file in test_images:
    # 파일명에서 숫자 부분 추출
    numbers = re.findall(r'\d+', img_file)
    # 이미지 ID 추출 로직 (가장 긴 숫자 시퀀스 사용)
    image_id = int(max(numbers, key=len)) if numbers else int(img_file.replace('.png', ''))
    test_image_mapping[img_file] = image_id

# YOLO 모델이 로드되었는지 확인
if 'model' in locals():
    for img_file in tqdm(test_images, desc="Inference"):
        img_path = os.path.join(test_img_dir, img_file)
        image_id = test_image_mapping[img_file]

        # 추론 실행
        results = model.predict(
            source=img_path,
            conf=0.25,  # 신뢰도 임계값
            iou=0.45,  # NMS IOU 임계값
            verbose=False,
        )

        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().item()
                cls = int(boxes.cls[i].cpu().item())

                # 제출 형식에 맞게 데이터 저장
                submission_data.append({
                    'annotation_id': annotation_id,
                    'image_id': image_id,
                    'category_id': cls + 1,  # 0-based index를 1-based category_id로 변환
                    'bbox_x': int(round(x1)),
                    'bbox_y': int(round(y1)),
                    'bbox_w': int(round(x2 - x1)),
                    'bbox_h': int(round(y2 - y1)),
                    'score': round(conf, 2)
                })
                annotation_id += 1

    # 저장
    submission_df = pd.DataFrame(submission_data)
    col_order = ['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score']
    submission_df = submission_df[col_order]

    submission_path = os.path.join(config['submission_dir'], 'submission_v1.csv')
    submission_df.to_csv(submission_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"제출 파일 생성 완료")
    print(f"{'=' * 60}")
    print(f"총 예측      : {len(submission_df):,}개")
    print(f"고유 이미지   : {submission_df['image_id'].nunique()}개")
    print(f"파일 위치    : {submission_path}")
    print(f"\n[미리보기]")
    print(submission_df.head(10))

else:
    print("ERROR: 모델 학습 또는 로드에 실패하여 추론을 건너뜁니다.")
