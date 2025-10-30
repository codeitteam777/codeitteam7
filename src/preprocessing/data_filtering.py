import pandas as pd
from constants import FILES_TO_EXCLUDE


def filter_master_data(master_data: dict) -> dict:
    """
    오류 파일 목록을 기반으로 master_data (학습 데이터)를 정제합니다.

    Args:
        master_data (dict): 이미지 파일명과 어노테이션 정보를 담은 딕셔너리.

    Returns:
        dict: 오류 파일이 제거된 정제된 master_data.
    """

    deleted_count = 0

    # 딕셔너리의 키를 미리 리스트로 복사하여 순회 중 삭제 오류를 방지합니다.
    for filename in list(master_data.keys()):
        if filename in FILES_TO_EXCLUDE:
            del master_data[filename]
            deleted_count += 1

    # 학습 시 로그를 위해 삭제된 파일 수를 반환할 수도 있습니다.
    # print(f"총 {deleted_count}개의 오류 파일을 삭제했습니다.")
    return master_data


# @title src/preprocessing/data_filtering.py (클린셋 추출 함수 추가)
# ... (기존 filter_master_data 함수 위/아래에 위치)

def extract_clean_master_data(master_data: dict, clean_files: list) -> dict:
    """
    master_data에서 clean_files 목록에 있는 유효한 파일만 추출하여 클린 데이터셋을 생성합니다.

    Args:
        master_data (dict): 전체 이미지 파일과 어노테이션 정보.
        clean_files (list): True Count와 Annotation Count가 일치하는 파일명 목록.

    Returns:
        dict: 유효한 파일만 포함된 클린 master_data.
    """
    clean_master_data = {}
    for filename in clean_files:
        if filename in master_data:
            clean_master_data[filename] = master_data[filename]

    return clean_master_data


def analyze_and_filter_data(master_data: dict) -> tuple:
    """
    파일명의 K-ID 개수와 JSON의 Annotation 개수를 비교하여 클린셋과 오류셋을 분리합니다.

    Args:
        master_data: 이미지 파일명을 키로 하고, 어노테이션 정보를 값으로 가진 딕셔너리.

    Returns:
        tuple: (클린셋 파일명 리스트, 오류 레코드 DataFrame)
    """
    error_records = []
    clean_files = []

    for img_filename, data in master_data.items():

        # 1. 파일명에서 "True Count" 파싱 로직
        # ... (여기에 해당 코드 블록의 파싱 로직을 구현)
        base_name = img_filename.split('_')[0]
        k_parts = base_name.split('-')
        true_count = len(k_parts) - 1

        # 2. 어노테이션에서 "Annotation Count" 파악
        annotation_count = len(data.get('annotations', []))

        # 3. 두 개수 비교
        if true_count == annotation_count:
            clean_files.append(img_filename)
        else:
            error_records.append({
                'filename': img_filename,
                'true_count': true_count,
                'annotation_count': annotation_count
            })

    error_df = pd.DataFrame(error_records)

    # 여기서 오류 파일을 제거하고 clean_files만 반환하도록 할 수 있습니다.
    return clean_files, error_df

def find_bbox_outliers(bbox_df, img_w=976, img_h=1280):
    """
    BBox 데이터를 받아 경계를 벗어나거나 크기가 비정상적인 BBox를 찾습니다.

    Args:
        bbox_df (pd.DataFrame): 모든 BBox 정보(x_center, w, h, area 등)를 담은 DataFrame.
        img_w (int): 이미지 너비.
        img_h (int): 이미지 높이.

    Returns:
        pd.DataFrame, pd.DataFrame: 경계 벗어난 BBox, 크기 이상치 BBox
    """
    # 1. 이미지 경계를 벗어나는 BBox 탐지
    out_of_bound = bbox_df[
        (bbox_df['x_center'] - bbox_df['w'] / 2 < 0) |
        (bbox_df['y_center'] - bbox_df['h'] / 2 < 0) |
        (bbox_df['x_center'] + bbox_df['w'] / 2 > img_w) |
        (bbox_df['y_center'] + bbox_df['h'] / 2 > img_h)
        ]

    # 2. 비정상적으로 작거나 큰 BBox 탐지 (IQR 기준)
    q1 = bbox_df['area'].quantile(0.25)
    q3 = bbox_df['area'].quantile(0.75)
    iqr = q3 - q1
    # 매우 작은 값 (0.1% 이하)을 가진 경우를 위해 lower_bound는 0 미만으로 내려가지 않게 조정할 수도 있음
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q1 + 1.5 * iqr

    size_outliers = bbox_df[
        (bbox_df['area'] < lower_bound) |
        (bbox_df['area'] > upper_bound)
        ]

    # 실제 학습에서는 이 함수를 호출하여 outlier의 'filename'과 'class_name'을 추출해
    # 해당 라벨을 제거하거나 이미지 전체를 데이터셋에서 제외하게 됩니다.
    return out_of_bound, size_outliers

# 만약 이 파일 자체를 실행해서 결과를 보고 싶다면:
if __name__ == "__main__":
    # master_data를 로드하는 로직 (JSON 파일을 읽어 master_data를 구성)
    # ...

    # clean_files, error_df = analyze_and_filter_data(master_data)
    # print(f"[클린셋] 개수 일치: {len(clean_files)}개")
    # print(f"[오류셋] 개수 불일치: {len(error_df)}개")
    pass  # 실제 실행 로직은 train.py에서 이 함수를 import해서 사용합니다.
