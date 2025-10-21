-- 간단한 해결책: 복합 키 사용
-- 실행 방법: psql -U postgres -d project1 -f create_tables_simple.sql
-- 기존 테이블 삭제
DROP TABLE IF EXISTS annotations CASCADE;

DROP TABLE IF EXISTS categories CASCADE;

DROP TABLE IF EXISTS images CASCADE;

-- images 테이블 (복합 PRIMARY KEY 사용)
CREATE TABLE images (
    id INTEGER NOT NULL,
    drug_N VARCHAR(50) NOT NULL,
    -- 약품 코드를 복합 키에 포함
    file_name VARCHAR(255) NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    imgfile VARCHAR(255),
    drug_S VARCHAR(100),
    back_color VARCHAR(100),
    drug_dir VARCHAR(50),
    light_color VARCHAR(50),
    camera_la NUMERIC(10, 2),
    camera_lo NUMERIC(10, 2),
    size INTEGER,
    dl_idx VARCHAR(50),
    dl_mapping_code VARCHAR(50),
    dl_name TEXT,
    dl_name_en TEXT,
    img_key TEXT,
    dl_material TEXT,
    dl_material_en TEXT,
    dl_custom_shape VARCHAR(100),
    dl_company TEXT,
    dl_company_en TEXT,
    di_company_mf TEXT,
    di_company_mf_en TEXT,
    item_seq BIGINT,
    di_item_permit_date VARCHAR(20),
    di_class_no TEXT,
    di_etc_otc_code VARCHAR(50),
    di_edi_code TEXT,
    chart TEXT,
    drug_shape VARCHAR(50),
    thick NUMERIC(10, 2),
    leng_long NUMERIC(10, 2),
    leng_short NUMERIC(10, 2),
    print_front VARCHAR(100),
    print_back VARCHAR(100),
    color_class1 VARCHAR(50),
    color_class2 VARCHAR(50),
    line_front VARCHAR(50),
    line_back VARCHAR(50),
    img_regist_ts VARCHAR(20),
    form_code_name VARCHAR(100),
    mark_code_front_anal TEXT,
    mark_code_back_anal TEXT,
    mark_code_front_img TEXT,
    mark_code_back_img TEXT,
    mark_code_front TEXT,
    mark_code_back TEXT,
    change_date VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, drug_N) -- 복합 PRIMARY KEY
);

-- categories 테이블
CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    supercategory VARCHAR(100),
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- annotations 테이블
CREATE TABLE annotations (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    area NUMERIC(15, 2),
    iscrowd INTEGER DEFAULT 0,
    bbox_x NUMERIC(10, 2),
    bbox_y NUMERIC(10, 2),
    bbox_width NUMERIC(10, 2),
    bbox_height NUMERIC(10, 2),
    ignore INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE -- image_id는 복합 키라 외래 키 설정 안함 (또는 (image_id, drug_N) 복합 외래 키 필요)
);

-- 인덱스 생성
CREATE INDEX idx_images_id ON images(id);

CREATE INDEX idx_images_drug_n ON images(drug_N);

CREATE INDEX idx_images_file_name ON images(file_name);

CREATE INDEX idx_images_dl_name ON images(dl_name);

CREATE INDEX idx_images_dl_company ON images(dl_company);

CREATE INDEX idx_annotations_image_id ON annotations(image_id);

CREATE INDEX idx_annotations_category_id ON annotations(category_id);

CREATE INDEX idx_categories_name ON categories(name);

COMMENT ON TABLE images IS '이미지 테이블 (id + drug_N 복합키로 여러 약품 지원)';
