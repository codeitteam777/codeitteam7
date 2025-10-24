import numpy as np

def prepare_stratified_split(clean_master_data: dict, class_count_lookup: dict) -> tuple[list, list]:
    """
    Stratified K-Fold/Split을 위한 파일명과 대표 레이블을 준비합니다.
    (이미지 내 가장 희귀한 클래스를 대표 레이블로 선정)

    Args:
        clean_master_data (dict): 정제된 이미지/어노테이션 정보.
        class_count_lookup (dict): 클래스 이름별 BBox 전체 개수 딕셔너리.

    Returns:
        tuple[list, list]: (X_filenames, y_stratify_labels)
    """
    x_filenames = []
    y_stratify_labels = []

    for filename, data in clean_master_data.items():
        if not data.get('annotations'):
            continue

        x_filenames.append(filename)

        # 이 이미지에 포함된 모든 클래스 이름
        img_classes = [ann['class_name'] for ann in data['annotations']]

        # 각 클래스의 전체 개수 (lookup 테이블 참조)
        # lookup에 없는 경우를 대비하여 매우 큰 값(예: 99999)으로 기본값을 설정합니다.
        counts = [class_count_lookup.get(name, 99999) for name in img_classes]

        # 개수가 가장 적은 클래스를 이 이미지의 대표로 선정
        rarest_class_name = img_classes[np.argmin(counts)]
        y_stratify_labels.append(rarest_class_name)

    return x_filenames, y_stratify_labels
