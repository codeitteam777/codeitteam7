# @title train.py
import os
import re
import shutil
import sys

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


# 프로젝트 루트(상위 디렉토리)를 sys.path에 추가
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, project_root)


from src.preprocessing.dataset import create_yolo_dataset_structure, get_cleaned_data_and_counts
from src.utils.data_split_utils import prepare_stratified_split
from src.utils.device import get_device

# from src.utils.data_split_utils import prepare_stratified_split

# --- 상수 Import: constants.py에서 필요한 모든 상수를 가져옵니다. ---
from constants import (
    PROJECT_ROOT, EXP_BASE_DIR, SUBMISSION_DIR, PROCESSED_DATA_PATH,
    YOLO_VER, TEST_IMG_DIR, TRAIN_IMG_DIR,
    TEST_SIZE, RANDOM_STATE, FILES_TO_EXCLUDE
)

# --- 프로젝트 모듈 Import ---
try:
    # InitDataset (src/data/init_dataset.py)을 로드합니다.
    from src.data.init_dataset import InitDataset
    # 기존에 논의된 전처리/유틸리티 모듈을 로드합니다.
    # from src.preprocessing.dataset import get_cleaned_data_and_counts, create_yolo_dataset_structure
    # from src.utils.data_split_utils import prepare_stratified_split
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("WARNING: Assuming helper functions are available for structure demonstration.")

# --------------------------------------------------------------------------------------
# 1. CONFIGURATION: 실험 하이퍼파라미터 정의 및 경로 구성
# --------------------------------------------------------------------------------------

# NOTE: 이 딕셔너리에는 실험별로 변경될 가능성이 높은 값만 남깁니다.
experiment_config = {
    # --- 학습 하이퍼파라미터 (YOLO) ---
    'epochs': 200,
    'imgsz': 640,
    'batch': 8,
    'device': 0,
    'project_name': 'yolo_v1',
    'patience': 20,
    'lr0': 0.01,
    'optimizer': 'AdamW',
    'augment': True,
    'close_mosaic': 10
}

# 상수를 사용하여 학습에 필요한 최종 경로를 구성합니다.
model_save_dir = os.path.join(EXP_BASE_DIR, "models")
submission_dir = SUBMISSION_DIR
yaml_path = os.path.join(PROCESSED_DATA_PATH, "yolo_dataset", "data.yaml")

# 폴더 생성
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

print("="*60)
print("경로 및 설정 완료")
print("="*60)


# --------------------------------------------------------------------------------------
# 2. 데이터 전처리 파이프라인 호출 (가장 크게 수정된 부분)
# --------------------------------------------------------------------------------------

try:
    print("\n[STEP 1/3] 원본 데이터 로드 및 통합 (init_dataset.py 호출)")
    init_dataset = InitDataset()
    # master_data와 class_to_id를 로드 및 생성
    master_data = init_dataset.integrate_data()
    class_to_id = init_dataset.class_to_id

    # 1단계: 데이터 정제 및 클래스 카운팅 (dataset.py에 정의된 함수 호출)
    print("\n[STEP 2/3] 오류 파일 제거 및 Stratified Split 준비")
    # FILES_TO_EXCLUDE 상수를 사용하여 오류 이미지 제거를 가정합니다.
    clean_master_data, class_count_lookup = get_cleaned_data_and_counts(
        master_data
    )

    # 2단계: Stratified Split을 위한 파일명 및 레이블 준비 (data_split_utils.py)
    x_filenames, y_stratify_labels = prepare_stratified_split(
        clean_master_data,
        class_count_lookup
    )

    # 3단계: YOLO 데이터셋 구축 (dataset.py)
    print("\n[STEP 3/3] YOLO 포맷 변환 및 데이터셋 구조 생성")
    yaml_path = create_yolo_dataset_structure(
        clean_master_data,
        x_filenames,
        y_stratify_labels,
        class_to_id,
        {
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
            # create_yolo_dataset_structure 함수가 사용할 경로/설정 전달
            'base_dir': PROJECT_ROOT,
            'train_img_dir': TRAIN_IMG_DIR
        }
    )
    print(f"\n데이터 준비 완료. YAML 경로: {yaml_path}")

except NameError as e:
    print(f"\n[ERROR] 모듈화된 전처리 함수를 찾을 수 없습니다: {e}")
    print("FATAL: 전처리 단계를 건너뛸 수 없습니다. 모듈 파일들을 확인해주세요.")
    sys.exit(1) # 오류 발생 시 스크립트 종료

# --------------------------------------------------------------------------------------
# 3. 모델 학습 (Train)
# --------------------------------------------------------------------------------------

print("\n학습 시작")
model = YOLO(YOLO_VER)

device = get_device()
model.to(device)

results = model.train(
    data=yaml_path,
    epochs=experiment_config['epochs'],
    imgsz=experiment_config['imgsz'],
    batch=experiment_config['batch'],
    device=device,
    project=model_save_dir,
    name=experiment_config['project_name'],
    exist_ok=True,
    patience=experiment_config['patience'],
    lr0=experiment_config['lr0'],
    optimizer=experiment_config['optimizer'],
    augment=experiment_config['augment'],
    close_mosaic=experiment_config['close_mosaic']
)

best_pt = os.path.join(model_save_dir, experiment_config['project_name'], 'weights', 'best.pt')
print(f"\n학습 완료 모델 저장: {best_pt}")

# --------------------------------------------------------------------------------------
# 4. 모델 검증 (Validation)
# --------------------------------------------------------------------------------------

print("\n모델 검증")
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

test_img_dir = TEST_IMG_DIR
test_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png')])

# 테스트 이미지 ID 매핑 생성
test_image_mapping = {}
for img_file in test_images:
    numbers = re.findall(r'\d+', img_file)
    image_id = int(max(numbers, key=len)) if numbers else int(img_file.replace('.png', ''))
    test_image_mapping[img_file] = image_id

if 'model' in locals():
    for img_file in tqdm(test_images, desc="Inference"):
        img_path = os.path.join(test_img_dir, img_file)
        image_id = test_image_mapping[img_file]

        results = model.predict(
            source=img_path,
            conf=0.25,
            iou=0.45,
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
                    'category_id': cls + 1,
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

    submission_path = os.path.join(submission_dir, 'submission_v1.csv')
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
