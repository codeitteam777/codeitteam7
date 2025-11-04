#!/usr/bin/env python3
"""
JSON 파일들을 PostgreSQL에 삽입 (복합 키 버전)
이미지 ID + 약품코드를 복합 PRIMARY KEY로 사용

사용법:
    python json_to_db_simple.py <json_files_directory>
"""

import os
import sys
import json
import psycopg2
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "project1"),
    "user": os.getenv("DB_USER", "magnum"),
    "password": os.getenv("DB_PASSWORD", ""),
}


class PillDatabaseManagerSimple:
    """약 데이터베이스 관리 클래스 (복합 키 버전)"""

    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.conn = None
        self.cursor = None

    def connect(self):
        """데이터베이스 연결"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            print(f"✓ 데이터베이스 연결 성공: {self.db_config['dbname']}")
        except psycopg2.Error as e:
            print(f"✗ 데이터베이스 연결 실패: {e}")
            sys.exit(1)

    def disconnect(self):
        """데이터베이스 연결 종료"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("✓ 데이터베이스 연결 종료")

    def insert_image(self, image_data: Dict[str, Any]) -> bool:
        """이미지 데이터 삽입 (복합 키 사용)"""
        query = """
        INSERT INTO images (
            id, drug_N, file_name, width, height, imgfile, drug_S, back_color,
            drug_dir, light_color, camera_la, camera_lo, size, dl_idx,
            dl_mapping_code, dl_name, dl_name_en, img_key, dl_material,
            dl_material_en, dl_custom_shape, dl_company, dl_company_en,
            di_company_mf, di_company_mf_en, item_seq, di_item_permit_date,
            di_class_no, di_etc_otc_code, di_edi_code, chart, drug_shape,
            thick, leng_long, leng_short, print_front, print_back,
            color_class1, color_class2, line_front, line_back, img_regist_ts,
            form_code_name, mark_code_front_anal, mark_code_back_anal,
            mark_code_front_img, mark_code_back_img, mark_code_front,
            mark_code_back, change_date
        ) VALUES (
            %(id)s, %(drug_N)s, %(file_name)s, %(width)s, %(height)s, %(imgfile)s,
            %(drug_S)s, %(back_color)s, %(drug_dir)s, %(light_color)s,
            %(camera_la)s, %(camera_lo)s, %(size)s, %(dl_idx)s,
            %(dl_mapping_code)s, %(dl_name)s, %(dl_name_en)s, %(img_key)s,
            %(dl_material)s, %(dl_material_en)s, %(dl_custom_shape)s,
            %(dl_company)s, %(dl_company_en)s, %(di_company_mf)s,
            %(di_company_mf_en)s, %(item_seq)s, %(di_item_permit_date)s,
            %(di_class_no)s, %(di_etc_otc_code)s, %(di_edi_code)s,
            %(chart)s, %(drug_shape)s, %(thick)s, %(leng_long)s,
            %(leng_short)s, %(print_front)s, %(print_back)s,
            %(color_class1)s, %(color_class2)s, %(line_front)s,
            %(line_back)s, %(img_regist_ts)s, %(form_code_name)s,
            %(mark_code_front_anal)s, %(mark_code_back_anal)s,
            %(mark_code_front_img)s, %(mark_code_back_img)s,
            %(mark_code_front)s, %(mark_code_back)s, %(change_date)s
        )
        ON CONFLICT (id, drug_N) DO UPDATE SET
            file_name = EXCLUDED.file_name,
            width = EXCLUDED.width,
            height = EXCLUDED.height,
            dl_name = EXCLUDED.dl_name,
            dl_company = EXCLUDED.dl_company,
            chart = EXCLUDED.chart,
            change_date = EXCLUDED.change_date
        """

        try:
            self.cursor.execute(query, image_data)
            return True
        except psycopg2.Error as e:
            print(
                f"✗ 이미지 삽입 실패 (ID: {image_data.get('id')}, drug_N: {image_data.get('drug_N')}): {e}"
            )
            return False

    def insert_category(self, category_data: Dict[str, Any]) -> bool:
        """카테고리 데이터 삽입"""
        query = """
        INSERT INTO categories (id, supercategory, name)
        VALUES (%(id)s, %(supercategory)s, %(name)s)
        ON CONFLICT (id) DO UPDATE SET
            supercategory = EXCLUDED.supercategory,
            name = EXCLUDED.name
        """

        try:
            self.cursor.execute(query, category_data)
            return True
        except psycopg2.Error as e:
            print(f"✗ 카테고리 삽입 실패 (ID: {category_data.get('id')}): {e}")
            return False

    def insert_annotation(self, annotation_data: Dict[str, Any]) -> bool:
        """어노테이션 데이터 삽입"""
        bbox = annotation_data.get("bbox", [0, 0, 0, 0])

        query = """
        INSERT INTO annotations (
            id, image_id, category_id, area, iscrowd,
            bbox_x, bbox_y, bbox_width, bbox_height, ignore
        ) VALUES (
            %(id)s, %(image_id)s, %(category_id)s, %(area)s, %(iscrowd)s,
            %(bbox_x)s, %(bbox_y)s, %(bbox_width)s, %(bbox_height)s, %(ignore)s
        )
        ON CONFLICT (id) DO UPDATE SET
            image_id = EXCLUDED.image_id,
            category_id = EXCLUDED.category_id,
            area = EXCLUDED.area
        """

        data = {
            "id": annotation_data.get("id"),
            "image_id": annotation_data.get("image_id"),
            "category_id": annotation_data.get("category_id"),
            "area": annotation_data.get("area"),
            "iscrowd": annotation_data.get("iscrowd", 0),
            "bbox_x": bbox[0] if len(bbox) > 0 else 0,
            "bbox_y": bbox[1] if len(bbox) > 1 else 0,
            "bbox_width": bbox[2] if len(bbox) > 2 else 0,
            "bbox_height": bbox[3] if len(bbox) > 3 else 0,
            "ignore": annotation_data.get("ignore", 0),
        }

        try:
            self.cursor.execute(query, data)
            return True
        except psycopg2.Error as e:
            print(f"✗ 어노테이션 삽입 실패 (ID: {annotation_data.get('id')}): {e}")
            return False

    def process_json_file(self, json_file_path: str) -> Dict[str, int]:
        """JSON 파일 처리"""
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return {"images": 0, "categories": 0, "annotations": 0, "errors": 1}

        counts = {"images": 0, "categories": 0, "annotations": 0, "errors": 0}

        # 카테고리 삽입
        if "categories" in data:
            categories = (
                data["categories"] if isinstance(data["categories"], list) else [data["categories"]]
            )
            for category in categories:
                if self.insert_category(category):
                    counts["categories"] += 1
                else:
                    counts["errors"] += 1

        # 이미지 삽입 (이제 모든 약품이 저장됨!)
        if "images" in data:
            images = data["images"] if isinstance(data["images"], list) else [data["images"]]
            for image in images:
                if self.insert_image(image):
                    counts["images"] += 1
                else:
                    counts["errors"] += 1

        # 어노테이션 삽입
        if "annotations" in data:
            annotations = (
                data["annotations"]
                if isinstance(data["annotations"], list)
                else [data["annotations"]]
            )
            for annotation in annotations:
                if self.insert_annotation(annotation):
                    counts["annotations"] += 1
                else:
                    counts["errors"] += 1

        return counts

    def process_directory(self, directory_path: str):
        """디렉토리 처리"""
        path = Path(directory_path)

        if not path.exists():
            print(f"✗ 경로가 존재하지 않습니다: {directory_path}")
            return

        if path.is_file():
            json_files = [path] if path.suffix == ".json" else []
        else:
            json_files = list(path.rglob("*.json"))

        if not json_files:
            print(f"✗ JSON 파일을 찾을 수 없습니다: {directory_path}")
            return

        print(f"\n총 {len(json_files)}개의 JSON 파일을 처리합니다.\n")

        total_counts = {"images": 0, "categories": 0, "annotations": 0, "errors": 0}

        for json_file in tqdm(json_files, desc="처리 중"):
            counts = self.process_json_file(str(json_file))

            for key in total_counts:
                total_counts[key] += counts[key]

            self.conn.commit()

        print("\n" + "=" * 60)
        print("처리 완료!")
        print("=" * 60)
        print(f"삽입된 이미지:      {total_counts['images']:>6}개 (모든 약품 포함!)")
        print(f"삽입된 카테고리:    {total_counts['categories']:>6}개")
        print(f"삽입된 어노테이션:  {total_counts['annotations']:>6}개")
        if total_counts["errors"] > 0:
            print(f"오류 발생:          {total_counts['errors']:>6}개")
        print("=" * 60)


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("사용법: python json_to_db_simple.py <json_files_directory>")
        sys.exit(1)

    json_path = sys.argv[1]

    print("=" * 60)
    print("약 데이터 PostgreSQL 삽입 (복합 키 버전)")
    print("=" * 60)

    db_manager = PillDatabaseManagerSimple(DB_CONFIG)

    try:
        db_manager.connect()
        db_manager.process_directory(json_path)
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
    finally:
        db_manager.disconnect()


if __name__ == "__main__":
    main()
