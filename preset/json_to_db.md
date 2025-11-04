# JSON => Database 실행 방법

```bash
# 현재 위치
cd preset

# table 삭제, table 생성
psql -U postgres -d project1 -f create_tables_simple.sql

# json => database
python json_to_db_simple.py 다운_받은_data가_있는_곳/train_annotations
```

# SQL Query

## 생성된 table

```sql
CREATE TABLE images ( ... );

CREATE TABLE categories ( ... );

CREATE TABLE annotations ( ... );
```

## category에 따른 이미지수

```sql
SELECT category_id, count(*) FROM annotations GROUP BY category_id ORDER by count(*);
```

## 이미지에 약 종류의 정보가 있는 갯수.

```sql
SELECT
	sum(CASE WHEN gc.cnt=4 THEN 1 else 0 end) AS cnt4,
	sum(CASE WHEN gc.cnt=3 THEN 1 else 0 end) AS cnt3,
    sum(CASE WHEN gc.cnt=2 THEN 1 else 0 end) AS cnt2,
    sum(CASE WHEN gc.cnt=1 THEN 1 else 0 end) AS cnt1
FROM (
	SELECT file_name, count(*) AS cnt FROM images GROUP BY file_name
) AS gc;
```

```
4: 489
3: 634
2: 302
1: 64
```


## 갯수에 해당하는 이미지

```sql
-- 이미지 1개
SELECT * FROM images INNER JOIN
(SELECT file_name, count(*) AS cnt FROM images GROUP BY file_name HAVING count(*) = 1) AS gc
ON gc.file_name = images.file_name;
```

<details>
<summary>이미지에 1개의 정보만 있는 파일 - 64개</summary>

```
K-001900-016548-019607-029345_0_2_0_2_70_000_200.png
K-001900-016548-021026-024850_0_2_0_2_90_000_200.png
K-001900-016551-018110-029345_0_2_0_2_90_000_200.png
K-001900-016551-019607-027926_0_2_0_2_90_000_200.png
K-001900-016551-021771-031705_0_2_0_2_70_000_200.png
K-001900-016551-027926-044199_0_2_0_2_70_000_200.png
K-002483-003743-006192-012081_0_2_0_2_70_000_200.png
K-002483-003743-012081-012778_0_2_0_2_70_000_200.png
K-002483-003743-012081-019552_0_2_0_2_70_000_200.png
K-002483-004378-005094-019552_0_2_0_2_90_000_200.png
K-002483-005094-019552-022362_0_2_0_2_75_000_200.png
K-002483-005094-022362-022627_0_2_0_2_90_000_200.png
K-002483-013395-019552-025438_0_2_0_2_75_000_200.png
K-003351-003832-016688_0_2_0_2_75_000_200.png
K-003351-003832-020238_0_2_0_2_70_000_200.png
K-003351-013900-016262_0_2_0_2_75_000_200.png
...
이하 생략
```
</details>

---

```sql
-- 이미지 2개.
SELECT * FROM images INNER JOIN
(SELECT file_name, count(*) AS cnt FROM images GROUP BY file_name HAVING count(*) = 2) AS gc
ON gc.file_name = images.file_name;
```

<details>
<summary>이미지에 2개의 정보만 있는 파일 - 302개</summary>

```
K-001900-010224-016551-031705_0_2_0_2_75_000_200.png
K-001900-010224-016551-033009_0_2_0_2_70_000_200.png
K-001900-010224-016551-033009_0_2_0_2_75_000_200.png
K-001900-016548-018110-021026_0_2_0_2_90_000_200.png
K-001900-016548-018110-029451_0_2_0_2_90_000_200.png
...
이하 생략
```
</details>
