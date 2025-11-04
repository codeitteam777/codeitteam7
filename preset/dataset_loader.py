# 운영체제에 따라 폰트 경로 설정
import platform
import os
import json
import glob
from tqdm import tqdm  # 진행 상황 표시
from collections import defaultdict, Counter
import pandas as pd  # 분석을 위해 pandas 사용
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
from matplotlib import font_manager, rc

if platform.system() == "Windows":
    # Windows: Malgun Gothic 설정
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc("font", family=font_name)
elif platform.system() == "Darwin":  # Mac OS
    # Mac OS: Apple Gothic 설정
    rc("font", family="AppleGothic")
else:
    # Linux 또는 기타 OS: 널리 사용되는 Nanum Gothic을 가정하거나, 기본 폰트 사용
    try:
        font_name = font_manager.FontProperties(
            fname="/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        ).get_name()
        rc("font", family=font_name)
    except:
        rc("font", family="DejaVu Sans")

# 마이너스 부호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False


# 파일 다운로드 개수 확인

base_dir = r"C:\Users\daboi\OneDrive\Desktop\ai05-level1-project"
base_dir = "/opt/data/codeit/project/data"

train_img_dir = os.path.join(base_dir, "train_images")
test_img_dir = os.path.join(base_dir, "test_images")
train_ann_dir = os.path.join(base_dir, "train_annotations")

# train_annotations 폴더 및 모든 하위 폴더에서 .json 파일 검색
json_files = glob.glob(os.path.join(train_ann_dir, "**", "*.json"), recursive=True)
print(f"총 {len(json_files)}개의 JSON 어노테이션 파일")

# train_images 폴더의 모든 이미지 파일 목록 수집
all_train_img_files = {f for f in os.listdir(train_img_dir) if not f.startswith(".")}
print(f"총 {len(all_train_img_files)}개의 학습 이미지 파일")
print(f"총 {len(os.listdir(test_img_dir))}개의 테스트 이미지 파일")

# 확인

with open(json_files[0], "r") as f:
    data = json.load(f)
print(data.keys())


# 파일 검증
# JSON에서 참조하는 이미지 파일명 수집
annotated_img_files = set()

for jf in json_files:
    try:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("images"):
            img_filename = data["images"][0].get("file_name")
            if img_filename:
                annotated_img_files.add(img_filename)
    except Exception as e:
        print(f"Error reading {jf}: {e}")
        continue

# 이미지 폴더에는 있는데 JSON이 참조하지 않는 이미지
imgs_without_annotations = all_train_img_files - annotated_img_files
print(f"어노테이션이 없는 이미지 수: {len(imgs_without_annotations)} 개")

# JSON은 참조하고 있는데 이미지 폴더에 실물 파일이 없는 경우
annotations_without_imgs = annotated_img_files - all_train_img_files
print(f"이미지가 없는 어노테이션 수: {len(annotations_without_imgs)} 개")

# 속성 확인

# 속성 확인
all_metadata_records = []
processing_errors = 0

# JSON 파일에서 추출할 메타데이터 속성 목록 (images 섹션의 모든 키를 동적으로 추출)
# 첫 번째 JSON 파일에서 모든 가능한 키를 추출하여 컬럼을 동적으로 생성
if json_files:
    with open(json_files[0], "r", encoding="utf-8") as f:
        sample_data = json.load(f)
        if sample_data and "images" in sample_data and sample_data["images"]:
            # file_name, width, height 등 약 50여개의 모든 속성 키를 추출
            all_possible_keys = list(sample_data["images"][0].keys())
        else:
            all_possible_keys = []
else:
    all_possible_keys = []

# 이미지당 캡처/약물 메타데이터 수집
for json_path in tqdm(json_files, desc="Collecting all metadata"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data.get("images"):
            processing_errors += 1
            continue

        img_info = data["images"][0]
        record = {}

        # 1. Image 및 Drug Metadata (약 50개 항목)
        for key in all_possible_keys:
            record[key] = img_info.get(key)

        # 2. Annotation Metadata (BBox 정보)
        # 한 JSON 파일에는 여러 개의 BBox가 있을 수 있으므로, 각 BBox마다 레코드를 생성합니다.

        if not data.get("annotations"):
            # 어노테이션이 없는 경우도 레코드에 포함 (필요하다면)
            record["bbox"] = None
            record["category_id"] = None
            all_metadata_records.append(record.copy())
            continue

        for ann in data["annotations"]:
            # BBox 정보 추가 (리스트 [x, y, w, h]로 저장)
            record["bbox"] = ann["bbox"]
            # BBox가 연결된 클래스 ID (dl_idx/dl_mapping_code와는 다름, categories 섹션의 id)
            record["annotation_category_id"] = ann["category_id"]

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
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
print(metadata_all_pills_df.head().to_string())

# 사용 후 옵션 복원
pd.reset_option("display.max_columns")
pd.reset_option("display.width")


# JSON 파일들을 읽어 하나의 데이터로 통합
master_data = defaultdict(lambda: {"image_path": "", "width": 0, "height": 0, "annotations": []})
class_to_id = {}
current_id = 0
processing_errors = 0

for json_path in json_files:

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 이미지 정보 추출
    img_info = data["images"][0]
    img_filename = img_info["file_name"]

    # 이미지 경로/크기 저장(이미지 당 1회)
    if not master_data[img_filename]["image_path"]:
        image_path = os.path.join(train_img_dir, img_filename)
        master_data[img_filename]["image_path"] = image_path
        master_data[img_filename]["width"] = img_info["width"]
        master_data[img_filename]["height"] = img_info["height"]

    # 클래스 이름 매핑 준비
    # 이 JSON 파일이 정의하는 카테고리(알약) 정보를 맵으로 만든다
    category_map = {cat["id"]: cat["name"] for cat in data["categories"]}

    # 어노테이션(BBox) 처리
    for ann in data["annotations"]:
        bbox = ann["bbox"]  # [x, y, w, h]

        # category_id를 사용해 실제 클래스 이름을 찾는다
        ann_cat_id = ann["category_id"]

        if ann_cat_id not in category_map:
            processing_errors += 1
            continue  # category_map에 없으면 이 알약(ann)은 건너뛰고 다음 알약으로

        class_name = category_map[ann_cat_id]

        # 클래스 ID 부여
        if class_name not in class_to_id:
            class_to_id[class_name] = current_id
            current_id += 1

        class_id = class_to_id[class_name]

        # 최종 어노테이션 추가
        master_data[img_filename]["annotations"].append(
            {
                "class_id": class_id,
                "class_name": class_name,  # 이름도 저장해두면 분석에 유용할 것 같다
                "bbox": bbox,
            }
        )

print(f"총 {len(master_data)}개의 이미지 데이터 처리")
print(f"총 {len(class_to_id)}개의 고유 클래스(알약 종류) 발견")

# 결과 저장
# 이게 진짜 학습 데이터이다 4526개의 JSON을 1489개의 이미지별 묶음으로 만들었다 이 안에 이미지의 모든 정답이 들어있다
with open(os.path.join(base_dir, "train_master_annotations.json"), "w", encoding="utf-8") as f:
    json.dump(master_data, f, ensure_ascii=False, indent=4)

# 이건 이름(키)와 숫자(value)를 연결해주는 맵이다
with open(os.path.join(base_dir, "class_to_id.json"), "w", encoding="utf-8") as f:
    json.dump(class_to_id, f, ensure_ascii=False, indent=4)

print("train_master_annotations.json 파일과 class_to_id.json 파일 저장")

# 삭제할 파일 목록 리스트업 (총 11개)
iou_error_files = [
    "K-003351-018147-020238_0_2_0_2_90_000_200.png",
    "K-003483-027733-030308-036637_0_2_0_2_90_000_200.png",
    "K-003351-020238-031863_0_2_0_2_70_000_200.png",
    "K-003351-029667-031863_0_2_0_2_70_000_200.png",
    "K-003483-019861-025367-029667_0_2_0_2_90_000_200.png",
    "K-002483-003743-012081-019552_0_2_0_2_90_000_200.png",
    "K-003483-019861-020238-031885_0_2_0_2_70_000_200.png",
    "K-003351-003832-029667_0_2_0_2_90_000_200.png",
    "K-001900-016548-019607-033009_0_2_0_2_70_000_200.png",
]

oob_error_files = [
    "K-003351-016262-018357_0_2_0_2_75_000_200.png",
    "K-003544-004543-012247-016551_0_2_0_2_70_000_200.png",
]

files_to_delete = iou_error_files + oob_error_files

# master_data에서 해당 파일들 삭제
deleted_count = 0
print(f"삭제 전 원본 master_data 개수: {len(master_data)}개")

for filename in files_to_delete:
    # master_data에 해당 키(파일명)가 있는지 확인
    if filename in master_data:
        # 딕셔너리에서 해당 항목 삭제
        del master_data[filename]
        deleted_count += 1
        # print(f"삭제 완료: {filename}") # 확인용 로그
    else:
        # 혹시 모르니 master_data에 파일이 없는 경우 로그
        print(f"경고: {filename}이 master_data에 없습니다.")

print(f"총 {deleted_count}개의 오류 파일을 삭제했습니다.")
print(f"정제 후 master_data 개수: {len(master_data)}개")

# 실제 train

# 시각화할 이미지 파일 경로 목록 생성
# master_data의 키(파일 이름)들을 리스트로 가져온다
all_filenames = list(master_data.keys())

# 전체 파일 목록 중 3개를 랜덤으로 선택
if len(all_filenames) >= 3:
    sample_filenames = random.sample(all_filenames, 3)
else:
    # 파일이 3개 미만일 경우 전체 파일을 사용
    sample_filenames = all_filenames

# image_path를 재구성
sample_image_paths = [os.path.join(train_img_dir, filename) for filename in sample_filenames]

# 이미지 로드 및 플로팅
plt.figure(figsize=(15, 6))
plot_index = 1

for i, img_path in enumerate(sample_image_paths):
    # OpenCV를 사용하여 이미지 로드
    img = cv2.imread(img_path)

    # 파일 로드 실패 검사 (안정성 확보)
    if img is None:
        print(f"파일 로드 실패 - '{img_path}'")
        continue

    # BGR -> RGB 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 서브플롯에 표시
    plt.subplot(1, 3, plot_index)
    plt.imshow(img_rgb)

    # 파일 이름에서 경로와 확장자 제거하여 제목으로 사용
    title = os.path.basename(img_path)
    plt.title(title, fontsize=12)
    plt.axis("off")

    plot_index += 1

plt.tight_layout()
plt.savefig("random_sample_images.png")
# plt.show()


# class_to_id의 key(이름)와 value(id)를 뒤집은 id_to_class 맵을 만든다
id_to_class = {v: k for k, v in class_to_id.items()}

# 시각화할 샘플 이미지 선정
# 알약 개수별로 이미지 파일명을 그룹화한다
images_by_count = defaultdict(list)
for img_filename, data in master_data.items():
    pill_count = len(data["annotations"])
    if pill_count > 0:  # 어노테이션이 1개 이상 있는 이미지만
        images_by_count[pill_count].append(img_filename)

# 알약 2개짜리, 4개짜리 이미지 파일명을 랜덤으로 하나씩 선택
try:
    sample_2_pill = random.choice(images_by_count[2])
    sample_4_pill = random.choice(images_by_count[4])
    sample_filenames = [sample_2_pill, sample_4_pill]
    print(f"시각화 샘플 선정 완료: {sample_filenames}")
except Exception as e:
    print(f"샘플 선정 오류: {e}. (2개 또는 4개짜리 이미지가 없을 수 있음)")
    sample_filenames = random.sample(list(master_data.keys()), 2)  # 대안: 그냥 2개 랜덤


# BBox 시각화 및 저장
plt.figure(figsize=(16, 10))

# BBox를 그릴 색상 (B, G, R)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Blue  # Green  # Red  # Yellow

for i, img_filename in enumerate(sample_filenames):
    img_path = os.path.join(train_img_dir, img_filename)

    # 이미지 로드 (OpenCV: BGR)
    image = cv2.imread(img_path)
    if image is None:
        print(f"오류: '{img_path}' 파일을 로드할 수 없음")
        continue

    # 이미지 -> RGB (Matplotlib 용)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # master_data에서 어노테이션 정보 가져오기
    annotations = master_data[img_filename]["annotations"]

    # 이미지에 BBox와 텍스트 그리기
    for j, ann in enumerate(annotations):
        # BBox 좌표 [x, y, w, h]
        x, y, w, h = map(int, ann["bbox"])

        # 클래스 ID로 이름 찾기
        class_id = ann["class_id"]
        class_name = id_to_class.get(class_id, "Unknown")

        # 색상 선택
        color = COLORS[j % len(COLORS)]  # BGR
        # RGB로 변환 (matplotlib에서 BGR로 그리면 색이 반전됨)
        color_rgb = (color[2] / 255, color[1] / 255, color[0] / 255)

        # OpenCV는 BGR 기준 (cv2.rectangle/putText 용)
        # Matplotlib은 RGB 기준 (plt.imshow 용)
        # 여기서는 image_rgb에 그리므로 RGB 색상을 사용

        # 사각형 그리기
        # cv2.rectangle(image, (startX, startY), (endX, endY), color, thickness)
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), color_rgb, thickness=3)

        # 텍스트 그리기 (cv2는 한글 처리가 복잡하므로 plt.text 사용)
        plt.subplot(1, 2, i + 1)
        plt.text(
            x,
            y - 10,
            class_name,
            color="black",
            fontsize=10,
            bbox=dict(facecolor=color_rgb, alpha=0.7, pad=0.1),
        )

    # 서브플롯에 이미지 표시
    plt.subplot(1, 2, i + 1)
    plt.imshow(image_rgb)
    plt.title(f"'{img_filename}'\n(Pills: {len(annotations)})", fontsize=12)
    plt.axis("off")

plt.tight_layout()
output_filename = "bbox_visualization_samples.png"
plt.savefig(output_filename)
print(f"'{output_filename}' 파일로 시각화 결과 저장")
# plt.show()


# 클래스 분포 분석 시작
# master_data에서 모든 클래스 이름(class_name)을 수집
all_class_names = []
for img_data in master_data.values():
    for ann in img_data["annotations"]:
        all_class_names.append(ann["class_name"])

print(f"총 {len(all_class_names)}개의 알약(= 바운딩 박스)")

# 클래스별 개수 카운트
class_counts = Counter(all_class_names)

# DataFrame으로 변환 및 정렬
class_df = pd.DataFrame(class_counts.items(), columns=["class_name", "count"])
class_df = class_df.sort_values(by="count", ascending=False).reset_index(drop=True)


print(f"(전체 {len(class_df)}개 클래스)")

# 전체 DataFrame 출력
print(class_df.to_string())

# 클래스 시각화
# 클래스 분포 막대 그래프 (Bar Plot) - (Using 'count')
TOP_N = 20  # 상위 20개 클래스만 시각화

plt.figure(figsize=(14, 6))
sns.barplot(
    x="class_name", y="count", data=class_df.head(TOP_N), palette="viridis"  # 'count' 컬럼 사용
)

plt.title(f"클래스별 알약(BBox) 개수 분포 (상위 {TOP_N}개)", fontsize=15)
plt.xlabel("클래스 이름", fontsize=12)
plt.ylabel("알약 개수 (BBox Count)", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("class_distribution_bar_plot_korean.png")
# plt.show()


# 누적 분포 그래프 (Cumulative Distribution Plot) - (Using 'count')
# 전체 데이터 중 상위 클래스가 얼마나 많은 비중을 차지하는지 확인
total_bbox_count = class_df["count"].sum()
class_df["cumulative_count"] = class_df["count"].cumsum()
class_df["cumulative_ratio"] = class_df["cumulative_count"] / total_bbox_count * 100

plt.figure(figsize=(14, 6))
# 누적 비율을 선 그래프로 표시
sns.lineplot(
    x=class_df.index + 1, y="cumulative_ratio", data=class_df, marker="o"  # 인덱스 + 1 = 순위
)

# 데이터 불균형의 심각도를 나타내는 주요 지점 표시
top_10_ratio = class_df.iloc[min(9, len(class_df) - 1)]["cumulative_ratio"]
top_20_ratio = class_df.iloc[min(19, len(class_df) - 1)]["cumulative_ratio"]

plt.axvline(x=10, color="r", linestyle="--", linewidth=1, label=f"Top 10: {top_10_ratio:.1f}%")
plt.axvline(x=20, color="orange", linestyle="--", linewidth=1, label=f"Top 20: {top_20_ratio:.1f}%")
plt.axhline(y=80, color="gray", linestyle=":", linewidth=1, label="80% 기준")

plt.title("클래스별 BBox 누적 점유율", fontsize=15)
plt.xlabel("클래스 순위 (Rank)", fontsize=12)
plt.ylabel("누적 비율 (%)", fontsize=12)
plt.grid(linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("class_cumulative_distribution_plot_korean.png")
# plt.show()


# 분석 대상 메타데이터 속성
TARGET_PROPERTIES = ["drug_shape", "color_class1", "print_front", "print_back", "dl_company_en"]

# 클래스별 속성값 저장을 위한 딕셔너리
# Key: 클래스 이름 (dl_name), Value: {속성: [값1, 값2, ...]}
class_metadata_agg = defaultdict(lambda: {prop: [] for prop in TARGET_PROPERTIES})

for json_path in tqdm(json_files, desc="Processing JSON metadata"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data["images"]:
            continue

        img_info = data["images"][0]

        # 클래스 이름 추출 (dl_name이 실제 알약 이름)
        class_name = img_info.get("dl_name")
        if not class_name:
            continue

        # 각 속성값 추출 및 저장
        for prop in TARGET_PROPERTIES:
            value = img_info.get(prop)
            if value is not None:
                class_metadata_agg[class_name][prop].append(value)

    except Exception as e:
        # print(f"Error processing {json_path}: {e}")
        continue

# 클래스별 메타데이터 고유값 카운트 분석
analysis_records = []

for class_name, props in class_metadata_agg.items():
    record = {"class_name": class_name}

    # 각 속성별 '고유한 값의 개수'를 계산
    for prop, values in props.items():
        # 고유값 개수 = len(set(values))
        record[f"unique_{prop}_count"] = len(set(values))

    # 해당 클래스가 전체 데이터셋에서 몇 번 등장했는지 (BBox가 아닌 JSON 파일 수 기준)
    record["json_count"] = len(
        props["drug_shape"]
    )  # 모든 JSON 파일은 drug_shape을 가질 것이므로 이를 기준으로 카운트
    analysis_records.append(record)

# DataFrame으로 변환
metadata_df = pd.DataFrame(analysis_records)

# 클래스 등장 횟수(json_count) 기준으로 내림차순 정렬
metadata_df = metadata_df.sort_values(by="json_count", ascending=False)

# 결과 출력 (상위 10개만 예시로 출력)
print("\n")
print("클래스별 메타데이터 고유값 개수 (상위 10개)")
print(metadata_df.head(10).to_string())

# 전체 클래스 메타데이터 분석 결과 CSV 저장 (전체 분석을 위해)
metadata_df.to_csv("class_metadata_uniqueness_analysis.csv", index=False, encoding="utf-8")
print("\n'class_metadata_uniqueness_analysis.csv' 파일 저장")


# 색상/모양 고정 클래스 비율 계산
print("데이터셋의 색상/모양 고정 클래스 비율")

total_classes = len(metadata_df)

# 모양(Shape)이 고정된 클래스 수
fixed_shape_count = len(metadata_df[metadata_df["unique_drug_shape_count"] == 1])
fixed_shape_ratio = fixed_shape_count / total_classes * 100

# 색상(Color)이 고정된 클래스 수
fixed_color_count = len(metadata_df[metadata_df["unique_color_class1_count"] == 1])
fixed_color_ratio = fixed_color_count / total_classes * 100

print(f"총 클래스 수: {total_classes}개")
print(f"- 모양(drug_shape)이 1개인 클래스: {fixed_shape_count}개 ({fixed_shape_ratio:.1f}%)")
print(f"- 색상(color_class1)이 1개인 클래스: {fixed_color_count}개 ({fixed_color_ratio:.1f}%)")


# 클래스 이상치 확인
variant_classes_df = metadata_df[
    (metadata_df["unique_drug_shape_count"] != 1) | (metadata_df["unique_color_class1_count"] != 1)
]

print(f"이상 클래스: {len(variant_classes_df)}개")
print(
    variant_classes_df[
        ["class_name", "json_count", "unique_drug_shape_count", "unique_color_class1_count"]
    ].to_string()
)


# 데이터의 다양성 인쇄 내용이 2가지 이상인 클래스 필터링
print("2가지 이상 인쇄 패턴을 가진 클래스")

# 앞면 또는 뒷면 인쇄가 1보다 큰 클래스 필터링
diverse_print_df = metadata_df[
    (metadata_df["unique_print_front_count"] > 1) | (metadata_df["unique_print_back_count"] > 1)
]

# 결과 출력
print(f"총 {len(diverse_print_df)}개의 클래스가 다양한 인쇄 패턴이 있음")
print(
    diverse_print_df[
        ["class_name", "json_count", "unique_print_front_count", "unique_print_back_count"]
    ]
    .head(5)
    .to_string()
)


# 인쇄 이상치 확인
# DataFrame으로 변환
metadata_all_pills_df = pd.DataFrame(all_metadata_records)
print(f"\n'metadata_all_pills_df' 생성 완료. (총 {len(metadata_all_pills_df)}개 레코드)")


# 메타데이터 논리 오류 탐색
# 데이터 준비
df = metadata_all_pills_df.copy()

# 'print_front', 'print_back', 'line_front', 'line_back'의 NaN을 빈 문자열('')로 대체
print_cols = ["print_front", "print_back", "line_front", "line_back"]
for col in print_cols:
    df[col] = df[col].fillna("")

# 논리 오류 조건 정의
# "앞면" 사진인데, '앞면 각인/줄'은 없고 '뒷면 각인/줄' 정보만 있음
cond1_dir = df["drug_dir"] == "앞면"
cond1_front_empty = (df["print_front"] == "") & (df["line_front"] == "")
cond1_back_filled = (df["print_back"] != "") | (df["line_back"] != "")

contradiction1 = df[cond1_dir & cond1_front_empty & cond1_back_filled]

# "뒷면" 사진인데, '뒷면 각인/줄'은 없고 '앞면 각인/줄' 정보만 있음
cond2_dir = df["drug_dir"] == "뒷면"
cond2_front_filled = (df["print_front"] != "") | (df["line_front"] != "")
cond2_back_empty = (df["print_back"] == "") & (df["line_back"] == "")

contradiction2 = df[cond2_dir & cond2_front_filled & cond2_back_empty]

# 결과 집계 및 출력
print("\n" + "=" * 70)
print("메타데이터 논리 오류 탐색 결과")
print("'앞면' 사진인데 '뒷면' 정보만 있는 경우: {:,} 건".format(len(contradiction1)))
print("'뒷면' 사진인데 '앞면' 정보만 있는 경우: {:,} 건".format(len(contradiction2)))

total_contradictions = len(contradiction1) + len(contradiction2)
print("-------------------------------------------------")
print(f"총 {total_contradictions:,} 건의 논리적 모순 데이터 발견")

if total_contradictions > 0:
    # .drop_duplicates()를 제거하여 리스트 타입 컬럼 문제 해결
    all_contradictions_df = pd.concat([contradiction1, contradiction2])

    print("\n" + "=" * 70)
    print("논리 오류가 가장 많이 발생한 클래스:")
    # 'dl_name' (클래스 이름) 기준으로 카운트
    print(all_contradictions_df["dl_name"].value_counts().to_string())

    print("\n" + "=" * 70)
    print("논리 오류가 발생한 이미지 파일:")
    # 'file_name' (이미지 파일) 기준으로 카운트
    print(all_contradictions_df["file_name"].value_counts().to_string())

else:
    print("\n논리적 모순 데이터 없음")


# 이미지 안에서 알약 중복 확인
has_duplicates = any(
    len([ann["class_name"] for ann in img_data["annotations"]])
    != len(set(ann["class_name"] for ann in img_data["annotations"]))
    for img_data in master_data.values()
)

print("한 이미지 내 동일 클래스 중복 존재:", "없음" if not has_duplicates else "있음")


# 이미지당 알약 개수 분포 분석 시작
# master_data에서 이미지별 알약 개수(ann_count)를 뽑아 리스트로 만든다
ann_counts_list = []
for img_data in master_data.values():
    ann_counts_list.append(len(img_data["annotations"]))

# 개수별로 카운트
count_distribution = Counter(ann_counts_list)

# DataFrame으로 변환
count_df = pd.DataFrame(count_distribution.items(), columns=["pill_count", "image_count"])
count_df = count_df.sort_values(by="pill_count", ascending=True).reset_index(drop=True)

print("이미지당 알약 개수 분포")
print(count_df.to_string())

# 데이터프레임 생성
# master_data에서 이미지별 알약 개수(ann_count)를 뽑아 리스트로 만든다
ann_counts_list = []
for img_data in master_data.values():
    ann_counts_list.append(len(img_data["annotations"]))

# 개수별로 카운트
count_distribution = Counter(ann_counts_list)

# DataFrame으로 변환
count_df = pd.DataFrame(count_distribution.items(), columns=["pill_count", "image_count"])
count_df = count_df.sort_values(by="pill_count", ascending=True).reset_index(drop=True)


# 시각화 (Bar Plot)
plt.figure(figsize=(8, 5))
ax = sns.barplot(x="pill_count", y="image_count", data=count_df, palette="Reds_d")

# 각 막대 위에 이미지 개수 텍스트 추가
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}개",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 9),
        textcoords="offset points",
        fontsize=11,
    )

plt.title("이미지당 알약 개수 분포", fontsize=15)
plt.xlabel("이미지 속 알약 개수", fontsize=12)
plt.ylabel("이미지 수", fontsize=12)
plt.xticks(fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("pill_count_distribution_bar_plot_clean.png")
# plt.show()

# 'K-ID' 개수를 저장할 리스트
true_counts_list = []
# 'K-ID' 패턴이 아닌 파일명을 저장할 리스트
filenames_without_k_ids = []

# master_data의 1,489개 이미지 파일명을 모두 순회
for img_filename in master_data.keys():

    try:
        # 파일명에서 "True Count"를 파싱

        # '_'를 기준으로 앞부분만 추출
        # 예: 'K-001900-016548-018110-027926'
        base_name = img_filename.split("_")[0]

        # 'K-' ID 패턴인지 확인
        if not base_name.startswith("K-"):
            filenames_without_k_ids.append(img_filename)
            continue  # K-ID 파일이 아니면 건너뜀

        # '-'를 기준으로 K-ID들을 분리
        # 예: ['K', '001900', '016548', '018110', '027926']
        k_parts = base_name.split("-")

        # 'K'를 제외한 ID 개수 (예: 5 - 1 = 4)
        true_count = len(k_parts) - 1

        true_counts_list.append(true_count)

    except Exception as e:
        # 혹시 모를 파싱 오류
        print(f"파싱 오류 발생: {img_filename}, 오류: {e}")

# "True Count"의 분포를 집계한다
if true_counts_list:
    # Counter를 사용해 개수별 빈도 계산
    count_distribution = Counter(true_counts_list)

    # DataFrame으로 변환
    count_df = pd.DataFrame(
        count_distribution.items(), columns=["알약 개수 (파일명 기준)", "이미지 수"]
    )
    count_df = count_df.sort_values(by="알약 개수 (파일명 기준)").reset_index(drop=True)

    print("\n" + "=" * 50)
    print("[코드 검증 결과]")
    print("파일명(True Count) 기준, 이미지별 알약 개수 분포")
    print("=" * 50)
    print(count_df.to_string(index=False))
else:
    print("\n[코드 검증 결과]")
    print("K-ID 패턴을 가진 이미지 파일을 찾을 수 없습니다.")

if filenames_without_k_ids:
    print("\n" + "=" * 50)
    print(f"참고: {len(filenames_without_k_ids)}개 파일은 K-ID 패턴이 아님")
    # 예시 5개만 출력
    print(filenames_without_k_ids[:5])


# 결과를 저장할 리스트
error_records = []
clean_files = []
missing_label_files = []

# master_data의 모든 이미지를 검사 (1489개)
for img_filename, data in master_data.items():

    # 1. 파일명에서 "True Count" 파싱
    # 'K-001900-016548-018110-027926_0_...png'

    # '_'를 기준으로 앞부분만 추출
    base_name = img_filename.split("_")[0]
    # 'K-001900-016548-018110-027926'

    # '-'를 기준으로 K-ID들을 분리
    k_parts = base_name.split("-")
    # ['K', '001900', '016548', '018110', '027926']

    # 'K'를 제외한 ID 개수
    true_count = len(k_parts) - 1

    # 2. 어노테이션에서 "Annotation Count" 파악
    annotation_count = len(data["annotations"])

    # 3. 두 개수 비교
    if true_count == annotation_count:
        # 두 개수가 일치 = 완벽한 데이터
        clean_files.append(img_filename)

    else:
        # 두 개수가 불일치 = 오류 데이터
        error_records.append(
            {
                "filename": img_filename,
                "true_count": true_count,  # 파일명 기준 개수
                "annotation_count": annotation_count,  # JSON 기준 개수
            }
        )

        if annotation_count < true_count:
            # 발견한 "레이블 누락" (예: 파일명 4개, 어노테이션 1개)
            missing_label_files.append(img_filename)
        # else:
        # (만약 있다면) 어노테이션이 더 많은 경우

# 결과 분석
print(f"총 {len(master_data)}개 이미지 검사 완료\n")
print("=" * 50)
print(f"[클린셋] 개수 일치: {len(clean_files)}개")
print(f"[오류셋] 개수 불일치: {len(error_records)}개")
print("=" * 50)

if error_records:
    # 오류 상세 내용 (상위 20개)
    error_df = pd.DataFrame(error_records)
    print("\n[오류 상세 내역 (개수 불일치 Top 20)]")
    print(error_df.head(20).to_string())

    # 오류 유형별 집계
    print("\n[오류 유형별 집계]")
    print(error_df.groupby(["true_count", "annotation_count"]).size())


# master_data에서 모든 BBox 정보를 수집
all_bboxes_data = []
ann_counts_map = {}

for img_filename, img_data in master_data.items():

    # 이미지당 알약 개수 미리 계산
    pill_count = len(img_data["annotations"])
    ann_counts_map[img_filename] = pill_count

    for ann in img_data["annotations"]:
        x, y, w, h = ann["bbox"]

        # BBox 속성 계산 및 리스트에 추가
        all_bboxes_data.append(
            {
                "filename": img_filename,
                "class_name": ann["class_name"],
                "w": w,
                "h": h,
                "area": w * h,
                "x_center": x + w / 2,
                "y_center": y + h / 2,
            }
        )

# DataFrame으로 변환
bbox_df = pd.DataFrame(all_bboxes_data)
# 이미지당 알약 개수 (pill_count) 정보 추가
bbox_df["pill_count"] = bbox_df["filename"].map(ann_counts_map)

plt.figure(figsize=(10, 6))
sns.boxplot(x="pill_count", y="area", data=bbox_df, palette="pastel")
plt.title("이미지당 알약 개수(pill_count)별 BBox 면적(area) 분포", fontsize=15)
plt.xlabel("이미지 속 알약 개수", fontsize=12)
plt.ylabel("BBox 면적 (w * h)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("pill_count_vs_area_boxplot.png")


# 이미지 속성 분석 시작
# master_data에서 이미지 크기 정보를 수집
image_properties = []
for img_filename, img_data in master_data.items():
    w = img_data["width"]
    h = img_data["height"]
    image_properties.append(
        {
            "filename": img_filename,
            "width": w,
            "height": h,
        }
    )

# DataFrame으로 변환
img_df = pd.DataFrame(image_properties)

print(f"총 {len(img_df)}개의 이미지 속성 분석")

# 이미지 크기(w, h) 통계
print("\n" + "=" * 50)
print("이미지 크기 (Width, Height) 통계")
# .describe()로 통계 확인
print(img_df[["width", "height"]].describe().to_string())

# 이미지 크기(w, h) 고유값 확인
print("\n" + "=" * 50)
print("이미지 크기 고유값")
# .groupby()로 (w, h) 조합별 개수 확인
size_counts = img_df.groupby(["width", "height"]).size().reset_index(name="count")
print(size_counts.to_string())


# 분석 대상 속성 (JSON 파일의 'images' 섹션에서 추출)
# back_color: 배경색, light_color: 조명색, camera_la/lo: 카메라 각도, 'size' 추가
CAPTURE_PROPERTIES = ["back_color", "light_color", "camera_la", "camera_lo", "drug_dir", "size"]

# 데이터를 통합할 딕셔너리
# Key: 이미지 파일 이름, Value: 해당 이미지의 캡처 속성
image_capture_agg = {}

for json_path in tqdm(json_files, desc="Processing JSON files"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data.get("images"):
            continue

        img_info = data["images"][0]
        img_filename = img_info["file_name"]

        # 이미지당 캡처 정보는 하나만 존재하므로, 중복 저장 방지
        if img_filename not in image_capture_agg:
            record = {}
            for prop in CAPTURE_PROPERTIES:
                # 속성값을 추출 (값이 없으면 'N/A'로 처리)
                record[prop] = img_info.get(prop, "N/A")

            image_capture_agg[img_filename] = record

    except Exception as e:
        # print(f"Error processing {json_path}: {e}") # 에러 메시지 출력은 생략
        continue

# DataFrame으로 변환
capture_df = pd.DataFrame.from_dict(image_capture_agg, orient="index")
capture_df = capture_df.reset_index().rename(columns={"index": "filename"})

# 캡처 환경 다양성 분석
# 고유값 개수 (다양성)
print("\n1. 캡처 환경 속성별 고유값 개수 (다양성)")
for prop in CAPTURE_PROPERTIES:
    print(f"- {prop:<10}: {capture_df[prop].nunique()}개의 고유값")

# 배경색 분포 (가장 중요한 편향 요소)
print("\n2. 배경색 (back_color) 분포 (Top 5)")
print(capture_df["back_color"].value_counts().head(5).to_string())

# 조명색 분포
print("\n3. 조명색 (light_color) 분포")
print(capture_df["light_color"].value_counts().head(5).to_string())

# 카메라 각도 통계 (수치형 속성)
print("\n4. 카메라 각도 및 크기 (camera_la, camera_lo, size) 통계")
# camera_la, camera_lo, size에 대해 describe()를 통해 통계 확인
print(capture_df[["camera_la", "camera_lo", "size"]].describe().to_string())


# BBox 분석 시작
IMG_W = 976
IMG_H = 1280

# master_data에서 모든 BBox 정보를 수집하고 DataFrame 생성
all_bboxes_data = []
ann_counts_map = {}

# master_data를 순회하며 데이터 추출
for img_filename, img_data in master_data.items():

    # 이미지당 알약 개수 미리 계산
    pill_count = len(img_data["annotations"])
    ann_counts_map[img_filename] = pill_count

    for ann in img_data["annotations"]:
        x, y, w, h = ann["bbox"]

        # BBox 속성 계산 및 리스트에 추가
        all_bboxes_data.append(
            {
                "filename": img_filename,
                "class_name": ann["class_name"],
                "w": w,
                "h": h,
                "area": w * h,
                # 위치 분석을 위한 중심 좌표 계산
                "x_center": x + w / 2,
                "y_center": y + h / 2,
            }
        )

# DataFrame으로 변환
bbox_df = pd.DataFrame(all_bboxes_data)
# 이미지당 알약 개수 (pill_count) 정보 추가
bbox_df["pill_count"] = bbox_df["filename"].map(ann_counts_map)
# BBox 절대 크기 통계 (Width, Height, Area)
print("=" * 60)
print("1. BBox 크기 (Width, Height, Area) 통계")
print(bbox_df[["w", "h", "area"]].describe().to_string())

# BBox 종횡비 분석 (Aspect Ratio: w / h)
bbox_df["aspect_ratio"] = bbox_df["w"] / bbox_df["h"]
print("\n" + "=" * 60)
print("2. BBox 종횡비 (Aspect Ratio) 통계")
print(bbox_df["aspect_ratio"].describe().to_string())

# 종횡비 분포
total_count = len(bbox_df)
square_ish = len(bbox_df[(bbox_df["aspect_ratio"] >= 0.9) & (bbox_df["aspect_ratio"] <= 1.1)])
portrait = len(bbox_df[bbox_df["aspect_ratio"] < 0.9])
landscape = len(bbox_df[bbox_df["aspect_ratio"] > 1.1])
print("\n" + "BBox 종횡비 대략적 분포")
print(f"정사각형/원형에 가까움 (0.9 ~ 1.1): {square_ish/total_count:.1%}")
print(f"세로로 김 (< 0.9): {portrait/total_count:.1%}")
print(f"가로로 김 (> 1.1): {landscape/total_count:.1%}")

# BBox 위치 분포 분석 (Normalized Center)
# 중심 좌표를 0에서 1 사이로 정규화
bbox_df["x_center_norm"] = bbox_df["x_center"] / IMG_W
bbox_df["y_center_norm"] = bbox_df["y_center"] / IMG_H
print("\n" + "=" * 60)
print("3. 정규화된 BBox 중심 좌표 (0~1) 통계")
# 평균이 (0.5, 0.5)에서 얼마나 벗어나는지 확인
print(bbox_df[["x_center_norm", "y_center_norm"]].describe().to_string())

# 이미지당 알약 개수와 BBox 평균 크기 관계 분석
# 이미지당 알약 개수 그룹별 BBox 면적 평균 계산
area_by_pill_count = bbox_df.groupby("pill_count")["area"].agg(["mean", "median", "std"])
print("\n" + "=" * 60)
print("4. 이미지당 알약 개수별 BBox 면적 통계")
# 알약 개수가 많아질수록 알약 하나의 평균 크기가 작아지는지 확인
print(area_by_pill_count.to_string(float_format="%.1f"))


# 데이터 준비
shape_map_df = metadata_all_pills_df[["dl_name", "drug_shape"]].drop_duplicates()

# 'bbox_df'와 'shape_map_df'를 병합 ('class_name'과 'dl_name' 기준)
cross_validation_df = pd.merge(
    bbox_df, shape_map_df, left_on="class_name", right_on="dl_name", how="left"
)

# 바로 여기서 '가바토파정 100mg' 같이 'drug_shape' 정보가 없는 데이터는 제외시켜준다
cross_validation_df = cross_validation_df.dropna(subset=["drug_shape"])

# 모양(Shape)별 종횡비(Aspect Ratio) 통계 분석
print("알약 모양(drug_shape)별 실제 BBox 종횡비(w/h) 통계")
shape_ar_stats = cross_validation_df.groupby("drug_shape")["aspect_ratio"].describe()
print(shape_ar_stats.to_string(float_format="%.3f"))

# 이상치 탐지: 논리적 모순
print("\n" + "=" * 70)
print("메타데이터와 실측값의 논리적 모순")

# '원형'인데 종횡비가 1.0에서 크게 벗어난 경우
outlier_circle = cross_validation_df[
    (cross_validation_df["drug_shape"] == "원형")
    & ((cross_validation_df["aspect_ratio"] < 0.7) | (cross_validation_df["aspect_ratio"] > 1.3))
]

# '장방형' 또는 '타원형'인데 종횡비가 1.0에 가까운 경우
outlier_rectangle = cross_validation_df[
    (cross_validation_df["drug_shape"].isin(["장방형", "타원형", "캡슐제"]))
    & (cross_validation_df["aspect_ratio"] >= 0.9)
    & (cross_validation_df["aspect_ratio"] <= 1.1)
]

print(f"\n  - 이상치 1 ('원형' 아님): {len(outlier_circle)}개 발견")
if len(outlier_circle) > 0:
    print("    -> 예시 클래스 (상위 5개):")
    print(outlier_circle["class_name"].value_counts().head(5).to_string())

print(f"\n  - 이상치 2 ('장방형/타원형' 아님): {len(outlier_rectangle)}개 발견")
if len(outlier_rectangle) > 0:
    print("    -> 예시 클래스 (상위 5개):")
    print(outlier_rectangle["class_name"].value_counts().head(5).to_string())

# 시각화 (Boxplot)
print("\n" + "=" * 70)
print("모양별 종횡비 분포")

plt.figure(figsize=(12, 6))
sns.boxplot(x="drug_shape", y="aspect_ratio", data=cross_validation_df)
plt.axhline(y=1.0, color="r", linestyle="--", label="Aspect Ratio 1.0 (정원형)")
plt.title("알약 모양(drug_shape)별 BBox 종횡비(Aspect Ratio) 분포", fontsize=15)
plt.xlabel("메타데이터 (모양)", fontsize=12)
plt.ylabel("실측 종횡비 (Width / Height)", fontsize=12)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

output_filename = "shape_vs_aspect_ratio_boxplot.png"
plt.savefig(output_filename)


# 바운딩 박스 이상치 탐지
def find_bbox_outliers(bbox_df, img_w=976, img_h=1280):
    # 이미지 경계를 벗어나는 바운딩 박스
    out_of_bound = bbox_df[
        (bbox_df["x_center"] - bbox_df["w"] / 2 < 0)
        | (bbox_df["y_center"] - bbox_df["h"] / 2 < 0)
        | (bbox_df["x_center"] + bbox_df["w"] / 2 > img_w)
        | (bbox_df["y_center"] + bbox_df["h"] / 2 > img_h)
    ]

    # 비정상적으로 작거나 큰 바운딩 박스 (IQR)
    Q1 = bbox_df["area"].quantile(0.25)
    Q3 = bbox_df["area"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    size_outliers = bbox_df[(bbox_df["area"] < lower_bound) | (bbox_df["area"] > upper_bound)]

    return out_of_bound, size_outliers, lower_bound, upper_bound


# 이상치 탐지 실행
out_of_bound, size_outliers, lower_bound, upper_bound = find_bbox_outliers(bbox_df)

print(f"이미지 경계 벗어난 바운딩 박스: {len(out_of_bound)}개")
print(f"크기 이상치 바운딩 박스: {len(size_outliers)}개")
print(f"이상치 기준 - 작은쪽: {lower_bound:.1f}, 큰쪽: {upper_bound:.1f}")


# 이상치 상세 분석
# 이미지 경계 벗어난 바운딩 박스 상세
if len(out_of_bound) > 0:
    print("이미지 경계를 벗어난 바운딩 박스:")
    for idx, row in out_of_bound.iterrows():
        print(f"  - 파일: {row['filename']}")
        print(f"    클래스: {row['class_name']}")
        print(f"    위치: ({row['x_center']:.1f}, {row['y_center']:.1f})")
        print(f"    크기: {row['w']}x{row['h']} (면적: {row['area']})")
else:
    print("이미지 경계를 벗어난 바운딩 박스 없음")

print("\n", "=" * 70)

# 크기 이상치 바운딩 박스 상세
if len(size_outliers) > 0:
    print(f"\n크기 이상치 바운딩 박스 ({len(size_outliers)}개):")

    # 너무 큰 바운딩 박스
    too_large = size_outliers[size_outliers["area"] > upper_bound]
    print(f"너무 큰 바운딩 박스: {len(too_large)}개")
    if len(too_large) > 0:
        print("대표적인 큰 바운딩 박스 클래스:")
        print(too_large["class_name"].value_counts().head(5))

    # 너무 작은 바운딩 박스
    too_small = size_outliers[size_outliers["area"] < lower_bound]
    print(f"너무 작은 바운딩 박스: {len(too_small)}개")
    if len(too_small) > 0:
        print("대표적인 작은 바운딩 박스 클래스:")
        print(too_small["class_name"].value_counts().head(5))


# BBox 분석 시작
# master_data에서 모든 BBox 정보를 수집하고 DataFrame 생성

# BBox 종횡비 분석 (Aspect Ratio: w / h)
bbox_df["aspect_ratio"] = bbox_df["w"] / bbox_df["h"]
# BBox 위치 분포 분석 (Normalized Center)
bbox_df["x_center_norm"] = bbox_df["x_center"] / IMG_W
bbox_df["y_center_norm"] = bbox_df["y_center"] / IMG_H


# IoU 계산 함수
# IoU 계산 함수 (Intersection over Union)
# BBox 형식: [x_center, y_center, w, h]를 사용하여 계산
def calculate_iou_from_center(box1, box2):
    # 형식: [xc, yc, w, h]
    xc1, yc1, w1, h1 = box1
    xc2, yc2, w2, h2 = box2

    # BBox를 [x_min, y_min, x_max, y_max] 형식으로 변환
    x_min1, y_min1 = xc1 - w1 / 2, yc1 - h1 / 2
    x_max1, y_max1 = xc1 + w1 / 2, yc1 + h1 / 2

    x_min2, y_min2 = xc2 - w2 / 2, yc2 - h2 / 2
    x_max2, y_max2 = xc2 + w2 / 2, yc2 + h2 / 2

    # 교차 영역 (Intersection) 좌표
    x_left = max(x_min1, x_min2)
    y_top = max(y_min1, y_min2)
    x_right = min(x_max1, x_max2)
    y_bottom = min(y_max1, y_max2)

    # 교차 영역 면적
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # 교차 영역이 없으면 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 합집합 영역 (Union) 면적
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    # IoU 계산
    return intersection_area / union_area


# IoU 계산 데이터 준비 및 실행
# 이미지별 BBox 쌍의 IoU 분석
all_iou_values = []
# master_data를 이미지별로 IoU 계산에 필요한 BBox 리스트로 변환
# { 'filename': [[x_center, y_center, w, h], ...], ... }
image_bboxes = defaultdict(list)

# 필요한 컬럼만 추출하여 BBox 리스트 생성 (순서 통일: [x_center, y_center, w, h])
for _, row in bbox_df.iterrows():
    # BBox는 [x_center, y_center, w, h] 형식으로 저장
    bbox_data = [row["x_center"], row["y_center"], row["w"], row["h"]]
    image_bboxes[row["filename"]].append(bbox_data)

# 이미지별 BBox 쌍 IoU 계산
for filename, bboxes in tqdm(image_bboxes.items(), desc="Calculating IoU"):
    num_bboxes = len(bboxes)
    if num_bboxes < 2:
        continue  # 객체가 1개 이하면 IoU 계산 불필요

    # 모든 BBox 쌍에 대해 IoU 계산
    for i in range(num_bboxes):
        for j in range(i + 1, num_bboxes):
            # 수정된 중심 좌표 기반 IoU 함수 사용 (통일된 형식)
            iou = calculate_iou_from_center(bboxes[i], bboxes[j])
            all_iou_values.append(iou)

iou_df = pd.Series(all_iou_values, name="IoU")

print("\n" + "=" * 60)

# IoU 분포 통계
print("\n1. IoU 값 분포 통계")
print(iou_df.describe().to_string())

# 2. 겹침 수준별 BBox 쌍 개수
# IoU 임계값: 0.5 이상은 '높은 겹침', 0.2 이상은 '중간 겹침'으로 분류
HIGH_OVERLAP_THRESHOLD = 0.5
MEDIUM_OVERLAP_THRESHOLD = 0.1

high_overlap = len(iou_df[iou_df >= HIGH_OVERLAP_THRESHOLD])
medium_overlap = len(
    iou_df[(iou_df >= MEDIUM_OVERLAP_THRESHOLD) & (iou_df < HIGH_OVERLAP_THRESHOLD)]
)
low_overlap = len(iou_df[iou_df < MEDIUM_OVERLAP_THRESHOLD])

total_pairs = len(iou_df)

print("\n2. BBox 겹침 수준별 개수")
print(f"총 BBox 쌍 개수: {total_pairs}개")
print(
    f"- 높은 겹침 (IoU >= {HIGH_OVERLAP_THRESHOLD}): {high_overlap}개 ({high_overlap / total_pairs:.1%})"
)
print(
    f"- 중간 겹침 ({MEDIUM_OVERLAP_THRESHOLD} <= IoU < {HIGH_OVERLAP_THRESHOLD}): {medium_overlap}개 ({medium_overlap / total_pairs:.1%})"
)
print(
    f"- 낮은 겹침 (IoU < {MEDIUM_OVERLAP_THRESHOLD}): {low_overlap}개 ({low_overlap / total_pairs:.1%})"
)


# IoU >= 0.2인 이미지 파일 탐지
# IoU 임계값을 0.2로 설정하여 중간 겹침까지 포함
# IoU >= 0.1인 이미지 파일 탐지 (임계값을 0.1로 변경)
HIGH_IOU_THRESHOLD = 0.1  # 이 줄만 0.2 → 0.1로 변경
high_iou_files = set()

# image_bboxes 딕셔너리를 사용하여 파일명 기반으로 탐색
for filename, bboxes in image_bboxes.items():
    num_bboxes = len(bboxes)
    if num_bboxes < 2:
        continue  # 객체가 1개 이하면 IoU 계산 불필요

    is_high_overlap = False

    # 모든 BBox 쌍에 대해 IoU 계산
    for i in range(num_bboxes):
        for j in range(i + 1, num_bboxes):

            # BBox 데이터 [x_center, y_center, w, h]
            xc1, yc1, w1, h1 = bboxes[i]
            xc2, yc2, w2, h2 = bboxes[j]

            # [x_min, y_min, x_max, y_max] 형식으로 변환
            x_min1, y_min1 = xc1 - w1 / 2, yc1 - h1 / 2
            x_max1, y_max1 = xc1 + w1 / 2, yc1 + h1 / 2
            x_min2, y_min2 = xc2 - w2 / 2, yc2 - h2 / 2
            x_max2, y_max2 = xc2 + w2 / 2, yc2 + h2 / 2

            # 교차 영역 (Intersection) 계산
            x_left = max(x_min1, x_min2)
            y_top = max(y_min1, y_min2)
            x_right = min(x_max1, x_max2)
            y_bottom = min(y_max1, y_max2)

            if x_right < x_left or y_bottom < y_top:
                iou = 0.0
            else:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - intersection_area
                iou = intersection_area / union_area

            # IoU가 임계값(0.1)을 초과하면 해당 파일명을 기록
            if iou >= HIGH_IOU_THRESHOLD:
                high_iou_files.add(filename)
                is_high_overlap = True
                break
        if is_high_overlap:
            break

print(f"\nIoU >= {HIGH_IOU_THRESHOLD}인 높은/중간 겹침 BBox를 포함하는 이미지 파일:")
for filename in high_iou_files:
    print(f"- {filename}")


# 9개 파일명 리스트로 변환
overlap_filenames = sorted(list(high_iou_files))

# BBox를 그릴 색상 (B, G, R) - 최대 4개
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Blue  # Green  # Red  # Yellow

# 3x3 그리드로 설정 (4x2 → 3x3으로 변경)
plt.figure(figsize=(20, 15))  # 사이즈도 약간 조정

for i, img_filename in enumerate(overlap_filenames):
    img_path = os.path.join(train_img_dir, img_filename)

    # 이미지 로드 (OpenCV: BGR)
    image = cv2.imread(img_path)
    if image is None:
        print(f"오류: '{img_path}' 로드 실패. 다음으로 넘어갑니다.")
        continue

    # 이미지 -> RGB (Matplotlib 용)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # master_data에서 어노테이션 정보 가져오기
    annotations = master_data[img_filename]["annotations"]

    # 서브플롯 생성 (4,2, i+1 → 3,3, i+1으로 변경)
    plt.subplot(3, 3, i + 1)

    # 이미지에 BBox와 텍스트 그리기
    for j, ann in enumerate(annotations):
        # BBox 좌표 [x, y, w, h]
        x, y, w, h = map(int, ann["bbox"])

        # 클래스 ID로 이름 찾기
        class_id = ann["class_id"]
        class_name = id_to_class.get(class_id, "Unknown")

        # 색상 선택 (Matplotlib RGB 0~1 스케일)
        color_bgr = COLORS[j % len(COLORS)]
        color_rgb = (color_bgr[2] / 255, color_bgr[1] / 255, color_bgr[0] / 255)

        # 사각형 그리기
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), color_rgb, thickness=3)

        # 텍스트 그리기 (j*20 : 텍스트가 겹치지 않게 y위치 살짝 조정)
        plt.text(
            x,
            y - 5 + (j * 20),
            class_name,
            color="black",
            fontsize=9,
            bbox=dict(facecolor=color_rgb, alpha=0.6, pad=0.1),
        )

    # 서브플롯에 이미지 표시
    plt.imshow(image_rgb)
    plt.title(f"{img_filename}", fontsize=10)
    plt.axis("off")

plt.tight_layout()
output_filename = "overlapping_bboxes_visualization.png"
plt.savefig(output_filename)


# 클린셋 추출
# clean_files 리스트를 사용

clean_master_data = {}

for filename in clean_files:
    # 1,478개의 master_data에서 clean_files 목록에 있는 파일만 골라서
    if filename in master_data:
        # clean_master_data 라는 새 딕셔너리에 복사
        clean_master_data[filename] = master_data[filename]

print(f"총 {len(master_data)}개 데이터 중,")
print(f"최종 클린 데이터셋 생성 완료: {len(clean_master_data)}개")


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class PillDataset(Dataset):
    """
    알약 Object Detection을 위한 Dataset 클래스

    Args:
        master_data: 이미지 파일명을 키로 하는 딕셔너리
                     각 값은 {'image_path', 'width', 'height', 'annotations'} 포함
        image_dir: 이미지가 저장된 디렉토리 경로
        transform: 이미지 전처리를 위한 torchvision transforms (선택)
    """

    def __init__(
        self,
        master_data,
        image_dir,
        transform=None,
        base_transform=None,
        strong_transform=None,
        minority_classes=None,
    ):
        self.master_data = master_data
        self.image_dir = image_dir
        # 하위 호환: 기존 transform은 기본 변환으로 사용
        self.base_transform = base_transform if base_transform is not None else transform
        self.strong_transform = strong_transform
        self.minority_classes = set(minority_classes or [])

        # 파일명 리스트 생성
        self.filenames = list(master_data.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_data = self.master_data[filename]

        # 이미지 로드
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")

        # 어노테이션 정보 추출
        annotations = img_data["annotations"]

        # BBox 좌표 추출: [x, y, w, h] -> [x_min, y_min, x_max, y_max]
        boxes = []
        labels = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            x_min, y_min = x, y
            x_max, y_max = x + w, y + h
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann["class_id"])

        # numpy array로 변환
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # 이미지 transform 적용 (카테고리 희소도에 따라 강/약 변환 선택)
        use_strong = self.strong_transform is not None and any(
            label in self.minority_classes for label in labels
        )
        if use_strong:
            image = self.strong_transform(image)
        elif self.base_transform:
            image = self.base_transform(image)

        # target 딕셔너리 생성 (Faster R-CNN 등의 모델 형식)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(
                [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes], dtype=torch.float32
            ),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        return image, target


print("PillDataset 클래스 정의 완료")


# 클래스 분포 계산 및 소수 클래스 집합 구성 (하위 35%)
class_counts = defaultdict(int)
for _fname, _item in clean_master_data.items():
    for _ann in _item.get("annotations", []):
        class_counts[_ann["class_id"]] += 1

counts_list = list(class_counts.values())
minority_classes = set()
if counts_list:
    threshold = float(np.percentile(counts_list, 35))
    minority_classes = {cid for cid, cnt in class_counts.items() if cnt < threshold}
print(f"클래스 분포 요약: 총 {len(class_counts)}개 클래스")
print(f"소수 클래스(하위 35%): {sorted(list(minority_classes))}")


# Transform 정의
base_transform = transforms.Compose(
    [
        transforms.ToTensor(),  # PIL Image -> Tensor [0, 1]
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

strong_transform = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomApply([transforms.RandomAutocontrast()], p=0.5),
        transforms.RandomApply([transforms.RandomEqualize()], p=0.3),
        transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.3),
        transforms.RandomApply([transforms.RandomPosterize(bits=4)], p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
    ]
)

# Dataset 생성 (clean_master_data 사용)
if "clean_master_data" not in globals():
    print("오류: 'clean_master_data'가 메모리에 없습니다. 위 셀들을 먼저 실행해주세요.")
    print("IoU 겹침 등을 제거한 클린 데이터셋이 필요합니다.")
else:
    pill_dataset = PillDataset(
        master_data=clean_master_data,
        image_dir=train_img_dir,
        base_transform=base_transform,
        strong_transform=strong_transform,
        minority_classes=minority_classes,
    )

    print(f"Clean Dataset 생성 완료: 총 {len(pill_dataset)}개 이미지")
    print(f"(원본 master_data에서 IoU 겹침 등이 제거된 클린 데이터)")

    # 샘플 데이터 확인
    sample_image, sample_target = pill_dataset[0]
    print(f"\n샘플 데이터 확인:")
    print(f"  - 이미지 shape: {sample_image.shape}")
    print(f"  - BBox 개수: {len(sample_target['boxes'])}")
    print(f"  - 클래스 레이블: {sample_target['labels'].tolist()}")
    print(f"  - BBox 좌표 (첫 번째): {sample_target['boxes'][0].tolist()}")


# Collate 함수 정의 (배치 처리를 위해 필요)
def collate_fn(batch):
    """
    Object Detection에서는 각 이미지마다 BBox 개수가 다르므로
    커스텀 collate 함수가 필요합니다.
    """
    return tuple(zip(*batch))


use_pin_memory = device == "cuda"

# DataLoader 생성
if "pill_dataset" in globals():
    # Train/Validation split (80:20)
    train_size = int(0.8 * len(pill_dataset))
    val_size = len(pill_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        pill_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # 재현성을 위한 시드 설정
    )

    # DataLoader 생성
    batch_size = 4  # 필요에 따라 조정
    num_workers = 0  # PicklingError 방지를 위해 0으로 설정 (멀티프로세싱 비활성화)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,  # GPU 사용 시 성능 향상
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
    )

    print(f"DataLoader 생성 완료:")
    print(f"  - Train set: {len(train_dataset)}개 이미지, {len(train_loader)}개 배치")
    print(f"  - Validation set: {len(val_dataset)}개 이미지, {len(val_loader)}개 배치")
    print(f"  - Batch size: {batch_size}")
    print(f"\nDataLoader 사용 예시:")
    print(f"  for images, targets in train_loader:")
    print(f"      # images: 배치 크기만큼의 이미지 튜플")
    print(f"      # targets: 배치 크기만큼의 타겟 딕셔너리 튜플")
    print(f"      pass")
else:
    print("오류: 'pill_dataset'이 생성되지 않았습니다. 위 셀을 먼저 실행해주세요.")


# DataLoader 동작 테스트
if "train_loader" in globals():
    print("DataLoader 배치 샘플 확인:\n")

    # 첫 번째 배치 가져오기
    images, targets = next(iter(train_loader))

    print(f"배치 크기: {len(images)}")
    print(f"\n각 이미지별 정보:")
    for i, (img, target) in enumerate(zip(images, targets)):
        print(
            f"  [{i}] 이미지 shape: {img.shape}, BBox: {len(target['boxes'])}개, 클래스: {target['labels'].tolist()}"
        )

    print(f"\n✓ DataLoader가 정상적으로 작동합니다!")
    print(f"✓ 학습에 사용할 준비가 완료되었습니다.")
else:
    print("오류: 'train_loader'가 생성되지 않았습니다.")
