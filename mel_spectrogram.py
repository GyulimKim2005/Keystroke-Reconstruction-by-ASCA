import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import librosa.display

# --- 경로 설정 ---
# 사용자님이 알려주신 wav 파일 경로
input_dir = r"C:\Users\gyulimkim\Desktop\Library\LAB\keyboard_acoustic_side_channel\src\wav\1_AULA_F87_Pro"
# 결과 이미지가 저장될 경로 (src/out/1_AULA_F87_Pro)
output_base_dir = os.path.join("out", "1_AULA_F87_Pro")

# --- 파라미터 설정 ---
n_fft = 1024
hop_length = 225
n_mels = 64
before = 1000
after = 9000
threshold_percentile = 95 # 상위 5% 에너지를 피크로 간주

def isolator(signal, sample_rate, n_fft, hop_length, before, after, threshold=None):
    """오디오 신호에서 키보드 타건음을 분리하는 함수"""
    strokes = []
    # STFT 계산 및 에너지 산출
    fft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    energy = np.sum(np.abs(fft)**2, axis=0)
    
    if threshold is None:
        threshold = np.percentile(energy, threshold_percentile)
        
    peaks = np.where(energy > threshold)[0]
    prev_end = sample_rate * 0.1 * (-1) # 중복 추출 방지용
    
    for i in range(len(peaks)):
        this_peak = peaks[i]
        timestamp = (this_peak * hop_length) + n_fft // 2
        
        # 이전 타건음과 0.1초 이상 간격이 있을 때만 추출
        if timestamp > prev_end + (0.1 * sample_rate):
            start = int(max(0, timestamp - before))
            end = int(min(len(signal), timestamp + after))
            keystroke = signal[start:end]
            
            # 길이 맞추기 (Padding)
            if len(keystroke) < (before + after):
                keystroke = np.pad(keystroke, (0, int((before + after) - len(keystroke))))
                
            strokes.append(keystroke)
            prev_end = timestamp + after
            
    return strokes

def main():
    print("--- WAV Batch Processing Started ---")
    
    # 1. 폴더 존재 확인
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    # 2. 결과 폴더 생성
    os.makedirs(output_base_dir, exist_ok=True)

    # 3. WAV 파일 목록 가져오기
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    
    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return

    print(f"Found {len(wav_files)} files. Processing...")

    for file_name in wav_files:
        file_path = os.path.join(input_dir, file_name)
        folder_name = os.path.splitext(file_name)[0] # 확장자 제외 파일명
        save_path = os.path.join(output_base_dir, folder_name)
        
        os.makedirs(save_path, exist_ok=True)

        try:
            # WAV 로드 (sr=None으로 원본 샘플링 레이트 유지)
            signal, sr = librosa.load(file_path, sr=None)
            
            # 타건음 분리
            strokes = isolator(signal, sr, n_fft, hop_length, before, after)
            
            if len(strokes) == 0:
                print(f"[-] {file_name}: No keystrokes detected. Check audio level.")
                continue

            # 멜-스펙트로그램 생성 및 저장
            for idx, stroke in enumerate(strokes):
                mel_spec = librosa.feature.melspectrogram(
                    y=stroke, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
                )
                mel_db = librosa.power_to_db(mel_spec, ref=np.max)

                plt.figure(figsize=(6, 4))
                librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length)
                plt.axis('off') # 축 제거
                
                # 파일 저장 예: out/1_AULA_F87_Pro/a/a_001.png
                output_file = os.path.join(save_path, f"{folder_name}_{idx+1:03d}.png")
                plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
                plt.close()

            print(f"[OK] {file_name}: Extracted {len(strokes)} strokes.")

        except Exception as e:
            print(f"[Error] Failed to process {file_name}: {e}")

    print("\n--- All Processes Completed Successfully ---")

if __name__ == "__main__":
    main()