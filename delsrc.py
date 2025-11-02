import shutil
import os

def read_with_auto_encoding(file_path):
    """
    파일을 여러 인코딩으로 시도해 읽기. 자동 인코딩 감지.
    
    Args:
        file_path (str): 입력 파일 경로
    
    Returns:
        list: 읽은 줄 리스트, 또는 None (실패 시)
    """
    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']  # 한국어 환경 우선
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            print(f"성공: 인코딩 '{encoding}'으로 파일 읽음.")
            return lines
        except UnicodeDecodeError:
            print(f"실패: '{encoding}' 인코딩으로 읽기 불가. 다음 시도...")
            continue
    
    # 최후 수단: latin-1 + errors='ignore' (바이트 무시)
    try:
        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
            lines = f.readlines()
        print("경고: 'latin-1' + 무시 모드로 읽음 (일부 문자 손실 가능).")
        return lines
    except Exception as e:
        print(f"최종 실패: {e}")
        return None

def remove_local_conflict_section(input_file, output_file, backup=True):
    """
    Git 병합 충돌에서 <<<<<<< HEAD ~ ======= 부분을 제거하는 함수.
    
    Args:
        input_file (str): 입력 파일 경로
        output_file (str): 출력 파일 경로
        backup (bool): 원본 백업 여부 (True 시 input_file.backup 생성)
    """
    if not os.path.exists(input_file):
        print(f"오류: 파일 '{input_file}'이 존재하지 않습니다.")
        return
    
    # 백업 생성
    if backup:
        backup_file = input_file + '.backup'
        shutil.copy2(input_file, backup_file)
        print(f"백업 생성: {backup_file}")
    
    # 자동 인코딩으로 파일 읽기
    lines = read_with_auto_encoding(input_file)
    if lines is None:
        print("파일 읽기 실패: 수동으로 인코딩 확인하세요.")
        return
    
    processed_lines = []
    in_local_section = False
    
    for line in lines:
        if line.strip().startswith('<<<<<<< HEAD'):
            in_local_section = True
            continue  # HEAD 줄 제거
        elif line.strip().startswith('======='):
            in_local_section = False
            # ======= 줄도 제거 (구분선이므로)
            continue
        elif in_local_section:
            # 로컬 섹션 내 줄 무시
            continue
        else:
            # 나머지 줄 유지 (원격 섹션, >>>>>>> 포함)
            processed_lines.append(line)
    
    # >>>>>>> 줄도 제거하려면 아래 주석 해제
    processed_lines = [line for line in processed_lines if not line.strip().startswith('>>>>>>>')]
    
    # 출력 시 UTF-8로 저장 (호환성 좋음)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)
    
    print(f"처리 완료: {output_file}")
    print("제거된 섹션 예시: 로컬(HEAD) 부분이 삭제되었습니다.")

# 사용 예시
if __name__ == "__main__":
    input_file = "delsrc.txt"  # 충돌 파일 경로 입력
    output_file = "delsrc-result.txt"      # 출력 파일 경로 입력
    remove_local_conflict_section(input_file, output_file)