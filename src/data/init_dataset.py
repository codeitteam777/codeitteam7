import sys
import os
import json
import glob
from tqdm import tqdm  # 진행 상황 표시
from collections import defaultdict, Counter
import pandas as pd # 분석을 위해 pandas 사용


# 프로젝트 루트(상위 디렉토리)를 sys.path에 추가
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.insert(0, project_root)

# 이제 import 가능
from constants import RAW_DATA_PATH, TEMP_DATA_PATH, TRAIN_IMG_DIR, TEST_IMG_DIR

class InitDataset:
    """데이터셋 초기화 및 메타데이터 수집 클래스"""

    def __init__(self):
        self.json_files = []
        self.init_files()
        self.validate_files()

    def get_dir_path(self):
        """
        데이터셋의 디렉토리 경로를 반환합니다.
        Args:
            train_img_dir, test_img_dir, train_ann_dir
        """

        source_dir = RAW_DATA_PATH

        train_img_dir = TRAIN_IMG_DIR
        test_img_dir = TEST_IMG_DIR
        train_ann_dir = os.path.join(source_dir, "train_annotations")

        return train_img_dir, test_img_dir, train_ann_dir

    def init_files(self):
        """데이터셋 초기화 및 메타데이터 수집"""

        source_dir = RAW_DATA_PATH

        train_img_dir, test_img_dir, train_ann_dir = self.get_dir_path()

        # train_annotations 폴더 및 모든 하위 폴더에서 .json 파일 검색
        self.json_files = glob.glob(os.path.join(train_ann_dir, "**", "*.json"), recursive = True)
        print(f"총 {len(self.json_files)}개의 JSON 어노테이션 파일")

        # train_images 폴더의 모든 이미지 파일 목록 수집
        self.all_train_img_files = {f for f in os.listdir(train_img_dir) if not f.startswith('.')}
        print(f"총 {len(self.all_train_img_files)}개의 학습 이미지 파일")
        print(f"총 {len(os.listdir(test_img_dir))}개의 테스트 이미지 파일")

        with open(self.json_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(data.keys())

        # 속성 확인
        all_metadata_records = []
        processing_errors = 0

        # JSON 파일에서 추출할 메타데이터 속성 목록 (images 섹션의 모든 키를 동적으로 추출)
        # 첫 번째 JSON 파일에서 모든 가능한 키를 추출하여 컬럼을 동적으로 생성
        if self.json_files:
            with open(self.json_files[0], 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
                if sample_data and 'images' in sample_data and sample_data['images']:
                    # file_name, width, height 등 약 50여개의 모든 속성 키를 추출
                    all_possible_keys = list(sample_data['images'][0].keys())
                else:
                    all_possible_keys = []
        else:
            all_possible_keys = []

        # 이미지당 캡처/약물 메타데이터 수집
        for json_path in tqdm(self.json_files, desc="Collecting all metadata"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if not data.get('images'):
                    processing_errors += 1
                    continue

                img_info = data['images'][0]
                record = {}

                # 1. Image 및 Drug Metadata (약 50개 항목)
                for key in all_possible_keys:
                    record[key] = img_info.get(key)

                # 2. Annotation Metadata (BBox 정보)
                # 한 JSON 파일에는 여러 개의 BBox가 있을 수 있으므로, 각 BBox마다 레코드를 생성
                if not data.get('annotations'):
                    # 어노테이션이 없는 경우도 레코드에 포함 (필요하다면)
                    record['bbox'] = None
                    record['category_id'] = None
                    all_metadata_records.append(record.copy())
                    continue

                for ann in data['annotations']:
                    # BBox 정보 추가 (리스트 [x, y, w, h]로 저장)
                    record['bbox'] = ann['bbox']
                    # BBox가 연결된 클래스 ID (dl_idx/dl_mapping_code와는 다름, categories 섹션의 id)
                    record['annotation_category_id'] = ann['category_id']

                    # 딕셔너리를 복사하여 리스트에 추가 (매 BBox마다 하나의 행)
                    all_metadata_records.append(record.copy())

            except Exception as e:
                processing_errors += 1
                # print(f"Error processing {json_path}: {e}")
                continue

        # DataFrame으로 변환
        metadata_all_pills_df = pd.DataFrame(all_metadata_records)

        print("\n" + "=" * 50)
        print(f"총 {len(metadata_all_pills_df)}개의 알약/BBox 레코드 수집 완료")
        print(f"처리 오류 발생 JSON 파일 수: {processing_errors}개")
        print(f"전체 속성(컬럼) 개수: {len(metadata_all_pills_df.columns)}개")

        # 데이터프레임의 첫 5행과 모든 컬럼을 출력
        print("\n첫 5개 알약 레코드의 모든 메타데이터:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(metadata_all_pills_df.head().to_string())

        # 사용 후 옵션 복원
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')

    def validate_files(self):
        """데이터셋 파일 유효성 검사"""
        # JSON에서 참조하는 이미지 파일명 수집
        annotated_img_files = set()

        for jf in self.json_files:
            try:
                with open(jf, 'r', encoding = 'utf-8') as f:
                    data = json.load(f)
                if data.get("images"):
                    img_filename = data["images"][0].get("file_name")
                    if img_filename:
                        annotated_img_files.add(img_filename)
            except Exception as e:
                print(f"Error reading {jf}: {e}")
                continue

        # 이미지 폴더에는 있는데 JSON이 참조하지 않는 이미지
        imgs_without_annotations = self.all_train_img_files - annotated_img_files
        print(f"어노테이션이 없는 이미지 수: {len(imgs_without_annotations)} 개")

        # JSON은 참조하고 있는데 이미지 폴더에 실물 파일이 없는 경우
        annotations_without_imgs = annotated_img_files - self.all_train_img_files
        print(f"이미지가 없는 어노테이션 수: {len(annotations_without_imgs)} 개")

    # 데이터 통합
    def integrate_data(self):
        """모든 JSON 파일의 데이터를 통합하여 이미지별로 묶음 처리"""

        train_img_dir, test_img_dir, train_ann_dir = self.get_dir_path()

        # JSON 파일들을 읽어 하나의 데이터로 통합
        master_data = defaultdict(lambda: {
            'image_path': '',
            'width': 0,
            'height': 0,
            'annotations': []
        })
        self.class_to_id = {}
        current_id = 0
        processing_errors = 0

        for json_path in self.json_files:

            with open(json_path, 'r', encoding = 'utf-8') as f:
                data = json.load(f)

            # 이미지 정보 추출
            img_info = data['images'][0]
            img_filename = img_info['file_name']

            # 이미지 경로/크기 저장(이미지 당 1회)
            if not master_data[img_filename]['image_path']:
                image_path = os.path.join(train_img_dir, img_filename)
                master_data[img_filename]['image_path'] = image_path
                master_data[img_filename]['width'] = img_info['width']
                master_data[img_filename]['height'] = img_info['height']

            # 클래스 이름 매핑 준비
            # 이 JSON 파일이 정의하는 카테고리(알약) 정보를 맵으로 만든다
            category_map = {cat['id']: cat['name'] for cat in data['categories']}

            # 어노테이션(BBox) 처리
            for ann in data['annotations']:
                bbox = ann['bbox'] # [x, y, w, h]

                # category_id를 사용해 실제 클래스 이름을 찾는다
                ann_cat_id = ann['category_id']

                if ann_cat_id not in category_map:
                    processing_errors += 1
                    continue # category_map에 없으면 이 알약(ann)은 건너뛰고 다음 알약으로

                class_name = category_map[ann_cat_id]

                # 클래스 ID 부여
                if class_name not in self.class_to_id:
                    self.class_to_id[class_name] = current_id
                    current_id += 1

                class_id = self.class_to_id[class_name]

                # 최종 어노테이션 추가
                master_data[img_filename]['annotations'].append({
                    'class_id': class_id,
                    'class_name': class_name, # 이름도 저장해두면 분석에 유용할 것 같다
                    'bbox': bbox
                })

        print(f"총 {len(master_data)}개의 이미지 데이터 처리")
        print(f"총 {len(self.class_to_id)}개의 고유 클래스(알약 종류) 발견")

        # 결과 저장
        # 이게 진짜 학습 데이터이다 4526개의 JSON을 1489개의 이미지별 묶음으로 만들었다 이 안에 이미지의 모든 정답이 들어있다
        with open(os.path.join(TEMP_DATA_PATH, "train_master_annotations.json"), "w", encoding = 'utf-8') as f:
            json.dump(master_data, f, ensure_ascii = False, indent = 4)

        # 이건 이름(키)와 숫자(value)를 연결해주는 맵이다
        with open(os.path.join(TEMP_DATA_PATH, "self.class_to_id.json"), "w", encoding = 'utf-8') as f:
            json.dump(self.class_to_id, f, ensure_ascii = False, indent = 4)

        print("train_master_annotations.json 파일과 self.class_to_id.json 파일 저장")

        return master_data

if __name__ == "__main__":

    # from src.data.init_dataset import InitDataset

    init_dataset = InitDataset()
    master_data = init_dataset.integrate_data()
