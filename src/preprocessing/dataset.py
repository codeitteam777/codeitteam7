# @title src/preprocessing/dataset.py
from collections import Counter
from data_filtering import filter_master_data
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# BBox 유틸리티와 상수를 import합니다.
from ..utils.bbox_utils import convert_to_yolo_format



# 주의: 이 파일에서는 JSON 파일을 로드하는 load_master_data 함수가
# 별도로 정의되어 있거나, master_data를 인수로 받는 상위 스크립트(train.py)에서
# 이 함수를 호출한다고 가정합니다.

def get_cleaned_data_and_counts(master_data: dict) -> tuple[dict, dict]:
    """
    원본 master_data를 받아 필터링하고, 클린셋의 클래스별 BBox 개수를 카운트하여 반환합니다.

    Args:
        master_data (dict): JSON 파일에서 로드된 원본 이미지/어노테이션 정보.

    Returns:
        tuple[dict, dict]: (오류 파일이 제거된 클린 master_data, 클래스별 BBox 카운트 딕셔너리)
    """

    print(f"[Data Preprocessing] 원본 데이터셋 크기: {len(master_data)}개")

    # 1. 데이터 필터링 실행
    # 이전에 data_filtering.py에 정의한 함수를 호출하여 오류 파일을 제거합니다.
    clean_master_data = filter_master_data(master_data)

    print(f"[Data Filtering] 정제된 데이터셋 크기: {len(clean_master_data)}개")

    # 2. 클린셋 기준 클래스 분포 분석 및 카운트
    clean_class_names = []

    for img_data in clean_master_data.values():
        for ann in img_data['annotations']:
            # 어노테이션에서 class_name을 추출하여 리스트에 추가
            clean_class_names.append(ann['class_name'])

    clean_class_counts = Counter(clean_class_names)

    # Stratified Split에 사용할 클래스별 전체 개수 Lookup Table
    class_count_lookup = dict(clean_class_counts)

    print(f"[Class Analysis] 정제 후 총 BBox 수: {len(clean_class_names):,}개")
    print(f"[Class Analysis] 고유 클래스 수: {len(class_count_lookup)}개")

    # 최종적으로 정제된 데이터와 카운트 정보를 반환
    return clean_master_data, class_count_lookup


# train.py에서 호출할 메인 함수
def create_yolo_dataset_structure(
    clean_master_data: dict,
    x_filenames: list,
    y_stratify_labels: list,
    class_to_id: dict,
    config: dict  # 환경 설정(경로, random_state, test_size 등)을 dict로 받습니다.
):
    # 1. Train/Val Split (Stratified)
    train_files, val_files = train_test_split(
        x_filenames,  # PEP 8에 따라 소문자 변수명 사용
        test_size=config['test_size'],
        stratify=y_stratify_labels,
        random_state=config['random_state']
    )
    print(f"Train/Val Split 완료. Train: {len(train_files)}개, Val: {len(val_files)}개")

    # 2. YOLO 폴더 구조 생성
    yolo_base_dir = os.path.join(config['base_dir'], "yolo_dataset")
    shutil.rmtree(yolo_base_dir, ignore_errors=True)  # 기존 폴더 삭제 후 재생성 (선택 사항)
    os.makedirs(yolo_base_dir, exist_ok=True)

    for split in ['train', 'val']:
        os.makedirs(os.path.join(yolo_base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_base_dir, split, 'labels'), exist_ok=True)

    # 3. 데이터셋 파일 생성 (내부 함수로 정의)
    def _copy_and_convert_data(file_list, split_name):
        for filename in tqdm(file_list, desc=f"Creating {split_name} dataset"):
            # img_data = clean_master_data[filename] # 이미 클린 데이터이므로 바로 사용
            img_data = clean_master_data[filename]

            # (나머지 이미지 복사 및 라벨 파일 생성 로직... (shutil.copy, open().write))
            src_img_path = os.path.join(config['train_img_dir'], filename)
            dst_img_path = os.path.join(yolo_base_dir, split_name, 'images', filename)
            shutil.copy(src_img_path, dst_img_path)

            label_filename = filename.replace('.png', '.txt')
            label_path = os.path.join(yolo_base_dir, split_name, 'labels', label_filename)

            with open(label_path, 'w') as f:
                for ann in img_data['annotations']:
                    class_id = ann['class_id']
                    bbox = ann['bbox']

                    x_center, y_center, width, height = convert_to_yolo_format(
                        bbox, img_data['width'], img_data['height']
                    )

                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    _copy_and_convert_data(train_files, 'train')
    _copy_and_convert_data(val_files, 'val')

    # 4. data.yaml 생성
    yaml_content = f"""# YOLO Dataset Configuration
path: {yolo_base_dir}
train: train/images
val: val/images

# Classes
nc: {len(class_to_id)}
names: {list(class_to_id.keys())}
"""

    yaml_path = os.path.join(yolo_base_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\nYOLO 데이터셋 및 data.yaml 생성 완료. 위치: {yolo_base_dir}")
    return yaml_path  # 학습 스크립트에서 yaml 경로를 바로 사용할 수 있도록 반환


if __name__ == "__main__":
    # 이 파일은 주로 다른 스크립트에서 import되어 사용되지만,
    # 직접 실행하여 테스트할 경우를 대비해 여기에 테스트 코드를 작성할 수 있습니다.
    print("dataset.py 모듈이 준비되었습니다.")
