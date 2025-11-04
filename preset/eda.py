"""
데이터셋 전처리 통합 스크립트
YOLOv8 학습을 위한 데이터셋을 준비합니다.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from tqdm import tqdm
import yaml
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys


def set_plot_font():
    try:
        from google.colab import drive  # noqa: F401

        path = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"  # 나눔 고딕
        font_name = fm.FontProperties(fname=path, size=10).get_name()  # 기본 폰트 사이즈 : 10
        plt.rc("font", family=font_name)

    except Exception:
        plt.rc("font", family=["Nanum Gothic", "DejaVu Sans"], size=8)


set_plot_font()


#  json 파일을 읽어서 pandas DataFrame으로 변환
def read_json_to_dataframe(json_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    디렉토리 내의 모든 JSON 파일을 읽어서 세 개의 pandas DataFrame으로 변환합니다.

    Args:
        json_dir (Path): JSON 파일들이 있는 디렉토리 경로

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: images, annotations, categories DataFrame
    """
    images_records = []
    annotations_records = []
    categories_records = []
    json_files = list(json_dir.glob("**/*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in directory: {json_dir}")

    for json_path in tqdm(json_files, desc="Reading JSON files"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            # Images
            for item in data["images"]:
                images_records.append(item)

            # Annotations
            for item in data["annotations"]:
                annotations_records.append(item)

            # Categories
            for item in data["categories"]:
                categories_records.append(item)

    images_df = pd.DataFrame(images_records)
    annotations_df = pd.DataFrame(annotations_records)
    categories_df = pd.DataFrame(categories_records)

    return images_df, annotations_df, categories_df


# 데이터 로드
path = "/opt/data/codeit/project/data/train_annotations"
json_dir = Path(path)
images_df, annotations_df, categories_df = read_json_to_dataframe(json_dir)

print("=== Images DataFrame ===")
print(images_df.head())
print(images_df.info())
print("\n=== Annotations DataFrame ===")
print(annotations_df.head())
print(annotations_df.info())
print("\n=== Categories DataFrame ===")
print(categories_df.head())
print(categories_df.info())


def check_bbox_anomalies(images_df: pd.DataFrame, annotations_df: pd.DataFrame) -> pd.DataFrame:
    """
    bbox 좌표가 이미지 범위를 벗어나는 이상치를 검사합니다.

    Args:
        images_df (pd.DataFrame): 이미지 정보가 담긴 DataFrame
        annotations_df (pd.DataFrame): annotation 정보가 담긴 DataFrame

    Returns:
        pd.DataFrame: 이상치가 발견된 데이터
    """
    # image_id를 기준으로 두 DataFrame 병합 - file_name 컬럼 추가
    merged_df = annotations_df.merge(
        images_df[["id", "width", "height", "file_name"]], left_on="image_id", right_on="id"
    )

    # bbox 좌표 분리
    merged_df["x"] = merged_df["bbox"].apply(lambda x: x[0])
    merged_df["y"] = merged_df["bbox"].apply(lambda x: x[1])
    merged_df["w"] = merged_df["bbox"].apply(lambda x: x[2])
    merged_df["h"] = merged_df["bbox"].apply(lambda x: x[3])

    # 이상치 조건 정의
    anomalies = merged_df[
        (merged_df["x"] < 0)
        | (merged_df["y"] < 0)
        | (merged_df["x"] + merged_df["w"] > merged_df["width"])
        | (merged_df["y"] + merged_df["h"] > merged_df["height"])
    ]

    # image_id 기준으로 중복 제거
    anomalies = anomalies.drop_duplicates(subset=["image_id"])

    return anomalies


# 이상치 검사 실행
anomalies = check_bbox_anomalies(images_df, annotations_df)

print("\n=== Bbox Anomalies ===")
print(f"발견된 이상치 개수: {len(anomalies)}")
if len(anomalies) > 0:
    print("\n이상치 데이터 샘플:")
    print(anomalies[["image_id", "bbox", "width", "height", "file_name"]].head())


def is_valid_bbox(obj) -> bool:
    """
    bbox가 유효한지 검사합니다.
    - 리스트/튜플/ndarray 이고 길이 >=4 이며 모든 값이 유한수인지 확인
    """
    if obj is None:
        return False
    if isinstance(obj, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(obj, dtype=float)
        except Exception:
            return False
        if arr.size < 4:
            return False
        return np.isfinite(arr[:4]).all()
    return False


def visualize_anomalies(anomalies: pd.DataFrame, images_dir: Path, max_images: int = 25):
    """
    이상치로 판별된 이미지들을 시각화합니다.
    - row에 x,y,w,h 컬럼이 없으면 bbox 리스트에서 추출
    - overlapping_bbox가 있으면 추가로 파란색 박스로 표시
    """
    # 동일 파일(또는 image_id) 중복 제거하여 한 번만 표시
    subset_key = (
        "file_name"
        if "file_name" in anomalies.columns
        else ("image_id" if "image_id" in anomalies.columns else None)
    )
    df_to_show = anomalies
    if subset_key is not None:
        df_to_show = anomalies.drop_duplicates(subset=[subset_key])

    n_images = min(len(df_to_show), max_images)
    if n_images == 0:
        print("시각화할 이상치 이미지가 없습니다.")
        return

    # Subplot 그리드 설정
    rows = int(np.ceil(n_images / 5))
    cols = min(n_images, 5)
    plt.figure(figsize=(10, 3 * rows))

    for idx, (_, row) in enumerate(df_to_show.head(n_images).iterrows()):
        # 파일명 안전 추출
        file_name = row.get("file_name") if "file_name" in row.index else None
        if not file_name:
            print(f"파일명이 없습니다. 행 건너뜀: image_id={row.get('image_id')}")
            continue

        img_path = images_dir / file_name
        if not img_path.exists():
            print(f"이미지를 찾을 수 없습니다: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"이미지 로드 실패: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img)

        # primary bbox 추출 (x,y,w,h 컬럼 우선, 없으면 bbox 리스트에서)
        if all(k in row.index for k in ["x", "y", "w", "h"]):
            x, y, w, h = row["x"], row["y"], row["w"], row["h"]
        else:
            bbox = row.get("bbox")
            if not is_valid_bbox(bbox):
                print(f"유효하지 않은 bbox: image_id={row.get('image_id')}")
                continue
            arr_bbox = np.asarray(bbox, dtype=float)
            x, y, w, h = arr_bbox[0], arr_bbox[1], arr_bbox[2], arr_bbox[3]

        # primary bbox 그리기 (빨간색, 두껍게)
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=3)
        ax.add_patch(rect)

        # overlapping_bbox가 있으면 파란색으로 추가 그리기 (안전 검사)
        ob = row.get("overlapping_bbox", None)
        if is_valid_bbox(ob):
            arr_ob = np.asarray(ob, dtype=float)
            ox, oy, ow, oh = arr_ob[0], arr_ob[1], arr_ob[2], arr_ob[3]
            rect2 = plt.Rectangle((ox, oy), ow, oh, fill=False, edgecolor="blue", linewidth=2)
            ax.add_patch(rect2)

        # 제목: image_id 및 가능하면 IoU 표시
        title = f"Image ID: {row.get('image_id')}"
        if "iou_value" in row.index and pd.notna(row["iou_value"]):
            title += f"\nIoU: {row['iou_value']:.3f}"
        ax.set_title(title)
        ax.axis("on")

    plt.tight_layout()
    plt.show()


# 이미지 디렉토리 설정 및 시각화
images_dir = Path(
    "/opt/data/codeit/project/data/train_images"
)  # 이미지 경로를 실제 경로로 수정해야 함
visualize_anomalies(anomalies, images_dir)


def check_bbox_size_ratio_anomalies(
    images_df: pd.DataFrame, annotations_df: pd.DataFrame
) -> pd.DataFrame:
    """
    bbox의 크기와 비율에 대한 이상치를 검사합니다.

    Args:
        images_df (pd.DataFrame): 이미지 정보가 담긴 DataFrame
        annotations_df (pd.DataFrame): annotation 정보가 담긴 DataFrame

    Returns:
        pd.DataFrame: 이상치가 발견된 데이터
    """
    merged_df = annotations_df.merge(
        images_df[["id", "width", "height", "file_name"]], left_on="image_id", right_on="id"
    )

    # bbox 좌표 분리 및 계산
    merged_df["x"] = merged_df["bbox"].apply(lambda x: x[0])
    merged_df["y"] = merged_df["bbox"].apply(lambda x: x[1])
    merged_df["w"] = merged_df["bbox"].apply(lambda x: x[2])
    merged_df["h"] = merged_df["bbox"].apply(lambda x: x[3])

    # bbox 면적 비율 계산
    merged_df["bbox_area"] = merged_df["w"] * merged_df["h"]
    merged_df["image_area"] = merged_df["width"] * merged_df["height"]
    merged_df["area_ratio"] = merged_df["bbox_area"] / merged_df["image_area"]

    # bbox 가로세로 비율 계산
    merged_df["aspect_ratio"] = merged_df["w"] / merged_df["h"]

    # 이상치 조건 정의
    anomalies = merged_df[
        # 면적이 너무 작은 경우 (1% 미만)
        (merged_df["area_ratio"] < 0.01)
        |
        # 면적이 너무 큰 경우 (90% 초과)
        (merged_df["area_ratio"] > 0.9)
        |
        # 가로세로 비율이 비정상적인 경우
        (merged_df["aspect_ratio"] > 5)
        | (merged_df["aspect_ratio"] < 0.2)  # 1/5
    ]

    # 이상 유형 표시
    anomalies["anomaly_type"] = ""
    anomalies.loc[anomalies["area_ratio"] < 0.01, "anomaly_type"] += "작은 크기/"
    anomalies.loc[anomalies["area_ratio"] > 0.9, "anomaly_type"] += "큰 크기/"
    anomalies.loc[anomalies["aspect_ratio"] > 5, "anomaly_type"] += "넓은 비율/"
    anomalies.loc[anomalies["aspect_ratio"] < 0.2, "anomaly_type"] += "긴 비율/"

    # 중복 제거
    anomalies = anomalies.drop_duplicates(subset=["image_id"])

    return anomalies


# 새로운 이상치 검사 실행
print("\n=== Size/Ratio Anomalies ===")
size_ratio_anomalies = check_bbox_size_ratio_anomalies(images_df, annotations_df)
print(f"발견된 이상치 개수: {len(size_ratio_anomalies)}")
if len(size_ratio_anomalies) > 0:
    print("\n이상치 데이터 샘플:")
    print(
        size_ratio_anomalies[
            ["image_id", "bbox", "area_ratio", "aspect_ratio", "anomaly_type"]
        ].head()
    )

# 새로운 이상치 시각화
print("\n크기/비율 이상치 이미지 시각화:")


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """두 bbox 간의 IOU(Intersection over Union)를 계산합니다."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 각 bbox의 영역 계산
    area1 = w1 * h1
    area2 = w2 * h2

    # 교차 영역 계산
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area


def find_overlapping_bboxes(
    images_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    min_iou: float = 0.0,
    max_iou: float = 1.0,
) -> pd.DataFrame:
    """
    동일한 이미지 내에서 서로 겹치는 bbox 쌍을 찾아 반환합니다.

    Args:
        images_df: COCO images DataFrame (id, width, height, file_name 등)
        annotations_df: COCO annotations DataFrame (image_id, bbox, category_id 등)
        min_iou: 최소 IoU 임계치 (포함)
        max_iou: 최대 IoU 임계치 (포함)

    Returns:
        pd.DataFrame: 각 행이 겹치는 한 쌍의 bbox. 컬럼: image_id, file_name, bbox,
        overlapping_bbox, iou_value, ann_id_1, ann_id_2, category_id_1, category_id_2
    """
    if annotations_df.empty:
        return pd.DataFrame(
            columns=[
                "image_id",
                "file_name",
                "bbox",
                "overlapping_bbox",
                "iou_value",
                "ann_id_1",
                "ann_id_2",
                "category_id_1",
                "category_id_2",
            ]
        )

    # file_name 등을 붙이기 위한 병합본 생성
    merged = annotations_df.merge(
        images_df[["id", "file_name", "width", "height"]],
        left_on="image_id",
        right_on="id",
        how="left",
    )

    results = []
    # 이미지별로 그룹핑하여 같은 이미지 내에서만 비교
    for image_id, group in merged.groupby("image_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < 2:
            continue
        for i in range(n - 1):
            bbox_i = group.loc[i, "bbox"]
            if not is_valid_bbox(bbox_i):
                continue
            for j in range(i + 1, n):
                bbox_j = group.loc[j, "bbox"]
                if not is_valid_bbox(bbox_j):
                    continue
                iou = calculate_iou(bbox_i, bbox_j)
                if (iou >= min_iou) and (iou <= max_iou):
                    results.append(
                        {
                            "image_id": image_id,
                            "file_name": group.loc[i, "file_name"],  # 동일 이미지
                            "bbox": bbox_i,
                            "overlapping_bbox": bbox_j,
                            "iou_value": float(iou),
                            "ann_id_1": (
                                group.loc[i, "id_x"]
                                if "id_x" in group.columns
                                else group.loc[i, "id"]
                            ),
                            "ann_id_2": (
                                group.loc[j, "id_x"]
                                if "id_x" in group.columns
                                else group.loc[j, "id"]
                            ),
                            "category_id_1": (
                                group.loc[i, "category_id"]
                                if "category_id" in group.columns
                                else None
                            ),
                            "category_id_2": (
                                group.loc[j, "category_id"]
                                if "category_id" in group.columns
                                else None
                            ),
                        }
                    )

    if not results:
        return pd.DataFrame(
            columns=[
                "image_id",
                "file_name",
                "bbox",
                "overlapping_bbox",
                "iou_value",
                "ann_id_1",
                "ann_id_2",
                "category_id_1",
                "category_id_2",
            ]
        )

    return pd.DataFrame(results)


# 겹치는 bbox 탐지 및 시각화 실행
print("\n=== Overlapping Bboxes (0.1 <= IoU <= 0.9) ===")
overlaps_df = find_overlapping_bboxes(images_df, annotations_df, min_iou=0.1, max_iou=0.9)
print(f"겹치는 bbox 쌍 개수: {len(overlaps_df)}")
if len(overlaps_df) > 0:
    # 파일별(없으면 image_id별) 가장 큰 IoU 한 건만 남김
    dedup = overlaps_df.sort_values("iou_value", ascending=False)
    if "file_name" in dedup.columns:
        dedup = dedup.drop_duplicates(subset=["file_name"])  # 파일당 1건
    elif "image_id" in dedup.columns:
        dedup = dedup.drop_duplicates(subset=["image_id"])  # 백업 키

    print("\n파일별 상위 IoU 샘플 (최대 25건):")
    cols_show = [
        c
        for c in ["image_id", "file_name", "iou_value", "ann_id_1", "ann_id_2"]
        if c in dedup.columns
    ]
    print(dedup[cols_show].head(25))

    print("\n겹침 시각화 (빨강=기준 bbox, 파랑=겹치는 bbox):")
    visualize_anomalies(dedup, images_dir, max_images=25)

# 겹치는 bbox 탐지 및 시각화 실행
print("\n=== Overlapping Bboxes (0.1 <= IoU <= 0.9) ===")
overlaps_df = find_overlapping_bboxes(images_df, annotations_df, min_iou=1, max_iou=9)
print(f"겹치는 bbox 쌍 개수: {len(overlaps_df)}")
if len(overlaps_df) > 0:
    # 파일별(없으면 image_id별) 가장 큰 IoU 한 건만 남김
    dedup = overlaps_df.sort_values("iou_value", ascending=False)
    if "file_name" in dedup.columns:
        dedup = dedup.drop_duplicates(subset=["file_name"])  # 파일당 1건
    elif "image_id" in dedup.columns:
        dedup = dedup.drop_duplicates(subset=["image_id"])  # 백업 키

    print("\n파일별 상위 IoU 샘플 (최대 25건):")
    cols_show = [
        c
        for c in ["image_id", "file_name", "iou_value", "ann_id_1", "ann_id_2"]
        if c in dedup.columns
    ]
    print(dedup[cols_show].head(25))

    print("\n겹침 시각화 (빨강=기준 bbox, 파랑=겹치는 bbox):")
    visualize_anomalies(dedup, images_dir, max_images=25)
