import os
import csv
import json

# 경로 설정
input_dir = r'Z:\LAB\keyboard_acoustic_side_channel\Keystrokes\Keystrokes\files'
output_dir = r'Z:\LAB\keyboard_acoustic_side_channel\Keystrokes\Keystrokes\extracted_IKI'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_files():
    # 1. 전체 파일 목록 가져오기
    all_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    total_files_count = len(all_files)
    
    success_count = 0
    error_count = 0

    print(f"총 {total_files_count}개의 파일 처리를 시작합니다.")

    for filename in all_files:
        output_name = filename.replace('.txt', '.jsonl')
        output_path = os.path.join(output_dir, output_name)
        
        try:
            # latin-1 인코딩으로 깨진 문자열 무시하고 강제 로드
            with open(os.path.join(input_dir, filename), 'r', encoding='latin-1') as f_in:
                # quoting=csv.QUOTE_NONE으로 따옴표 에러(EOF) 원천 차단
                reader = csv.DictReader(f_in, delimiter='\t', quoting=csv.QUOTE_NONE)
                
                rows = []
                for row in reader:
                    try:
                        # 필수 데이터가 있는 행만 추출
                        if 'PRESS_TIME' in row and 'LETTER' in row and row['PRESS_TIME'] and row['LETTER']:
                            rows.append({
                                'time': float(row['PRESS_TIME']),
                                'letter': str(row['LETTER']) if row['LETTER'] != ' ' else 'SPACE'
                            })
                    except:
                        continue # 비정상 행은 해당 행만 스킵

                # 유효 데이터가 너무 적으면 에러로 간주
                if len(rows) < 2:
                    raise ValueError("Insufficient valid data rows")

                rows.sort(key=lambda x: x['time'])
                unique_keys = sorted(list(set(r['letter'] for r in rows)))
                label_map = {key: i for i, key in enumerate(unique_keys)}

                # 가공된 데이터 저장
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(f"classes: {len(unique_keys)}\n")
                    f_out.write(f"label_map: {label_map}\n")
                    f_out.write("Edges:\n")
                    for i in range(len(rows) - 1):
                        iki_sec = (rows[i+1]['time'] - rows[i]['time']) / 1000.0
                        f_out.write(f"{i} -> {i+1} attr={max(0.0, iki_sec)}\n")
                
                success_count += 1

        except Exception:
            # 파일 자체가 심하게 깨졌거나 읽을 수 없는 경우
            error_count += 1

        # 2. 실시간 통계 계산 및 출력 (100개 파일마다)
        processed_so_far = success_count + error_count
        if processed_so_far % 100 == 0 or processed_so_far == total_files_count:
            error_rate = (error_count / processed_so_far) * 100
            print(f"진행률: {processed_so_far}/{total_files_count} "
                  f"| 성공: {success_count} "
                  f"| 에러(미스): {error_count} "
                  f"| 현재 에러율: {error_rate:.2f}%")

    print("\n" + "="*50)
    print(f"처리 완료")
    print(f"Total: {total_files_count} | Success: {success_count} | Errors: {error_count}")
    print(f"최종 에러 발생률: {(error_count / total_files_count) * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    process_files()