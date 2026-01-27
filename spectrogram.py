import librosa
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

def calculate_global_energy(y, sr):
    """
    전체 주파수 대역의 에너지를 합산하여 피크 탐지용 신호를 만듭니다.
    """
    stft = np.abs(librosa.stft(y, n_fft=1024))
    return np.sum(stft, axis=0)

def detect_keystrokes(energy, y, sr, hop_length=512):
    """
    Press/Release 중복을 방지하면서 13개의 키 입력을 정확히 추출합니다.
    """
    # 임계값: 평균의 1.8배 (이미지가 선명하므로 이 정도가 적당합니다)
    threshold = np.mean(energy) * 1.8 
    
    # [수정] 15개 이상의 중복 탐지를 막기 위해 간격을 0.18초로 상향 조정합니다.
    min_dist = int(0.18 * sr / hop_length) 
    
    buffer_samples = int(0.35 * sr)
    safe_end_frame = (len(y) - buffer_samples) // hop_length
    
    peaks = []
    last_peak_sample = -int(0.3 * sr) # 중복 방지를 위한 이전 피크 위치 저장
    
    for i in range(min_dist, safe_end_frame):
        sample_idx = i * hop_length
        
        # 에너지 조건 + 피크 조건 + 파일 끝부분 안전 거리 확인
        if (energy[i] > threshold and 
            energy[i] == np.max(energy[i-min_dist:i+min_dist]) and
            sample_idx < len(y) - buffer_samples):
            
            # [핵심] 0.18초 이내에 발생하는 추가 피크(Release 소음)는 무시합니다.
            if sample_idx - last_peak_sample > int(0.18 * sr):
                peaks.append(sample_idx)
                last_peak_sample = sample_idx
    
    return peaks

def save_linear_spectrograms(y, peaks, sr, output_dir):
    duration = int(0.33 * sr)
    
    # 파일 전체에서 가장 큰 진폭을 찾아 기준(ref)으로 삼습니다.
    # 이렇게 하면 파일마다 선명도가 들쭉날쭉하지 않고 통일됩니다.
    global_max = np.max(np.abs(y))
    
    for i, start in enumerate(peaks):
        cut_start = max(0, start - int(0.05 * sr))
        y_segment = y[cut_start : cut_start + duration]
        
        if len(y_segment) < duration:
            y_segment = np.pad(y_segment, (0, duration - len(y_segment)))

        # 리니어 스펙트로그램 계산 (n_mels 없이 주파수 대역을 그대로 사용)
        stft = np.abs(librosa.stft(y_segment, n_fft=1024, hop_length=500))
        
        # global_max를 기준으로 사용하여 선명도를 통일합니다.
        spec_db = librosa.amplitude_to_db(stft, ref=global_max)
        
        final_img = cv2.resize(spec_db, (64, 64))
        plt.imsave(f"{output_dir}/spec_{i+1:02d}.png", final_img, cmap='magma')
        print(f"  > [Save] spec_{i+1:02d}.png (at {start/sr:.2f}s)")

def run_pipeline(audio_path):
    output_dir = "extracted_spectrograms"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    sr = 44100
    y, _ = librosa.load(audio_path, sr=sr)
    
    energy = calculate_global_energy(y, sr)
    peaks = detect_keystrokes(energy, y, sr)
    
    print(f"[*] 분석 결과: 총 {len(peaks)}개의 키 입력을 찾았습니다.")
    save_linear_spectrograms(y, peaks, sr, output_dir)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/spectrogram.py <audio_file>")
    else:
        run_pipeline(sys.argv[1])
        print("[*] 모든 이미지가 'extracted_spectrograms' 폴더에 저장되었습니다.")