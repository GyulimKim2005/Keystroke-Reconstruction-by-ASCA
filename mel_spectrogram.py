import librosa
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

def run_pipeline(audio_path):
    output_dir = "extracted_keys"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # 1. 오디오 로드 (표준 44.1kHz)
    sr = 44100
    y, _ = librosa.load(audio_path, sr=sr)
    total_samples = len(y)
    
    # 2. 순수 에너지 기반 피크 탐지 (기존 연구 방식)
    # STFT를 통해 시간별 전체 에너지를 계산합니다.
    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    energy = np.sum(stft, axis=0)
    
    # 임계값: 평균의 1.5배 (선명한 파일이라면 이 정도로 충분합니다)
    threshold = np.mean(energy) * 1.5
    
    # [수정] Press와 Release가 중복 탐지되지 않도록 간격을 0.15초로 소폭 늘림 (17개 -> 13개 유도)
    min_dist = int(0.15 * sr / 512) 
    
    # 파일 끝부분 검은 이미지 생성을 막기 위한 안전 경계
    safe_limit = total_samples - int(0.4 * sr)
    
    peaks = []
    last_peak_sample = -int(0.2 * sr) # 중복 방지를 위한 마지막 피크 위치 저장
    
    for i in range(min_dist, len(energy) - min_dist):
        sample_idx = i * 512
        
        # 에너지 조건을 만족하고, 이전 피크와 최소 간격이 유지되며, 파일 끝자락이 아닐 때만 추가
        if (energy[i] > threshold and 
            energy[i] == np.max(energy[i-min_dist:i+min_dist]) and
            sample_idx < safe_limit):
            
            # [추가] 롤오버 및 Press/Release 중복 방지 로직
            if sample_idx - last_peak_sample > int(0.5 * sr): # 키 입력 간 0.5초 제한 설정
                peaks.append(sample_idx)
                last_peak_sample = sample_idx
            
    print(f"[*] 총 {len(peaks)}개의 키 입력을 감지했습니다.")

    # 3. 멜-스펙트로그램 가공 및 저장 (Standard Setting)
    duration = int(0.33 * sr) # 0.33초
    for i, start in enumerate(peaks):
        # 타격 순간을 포착하기 위해 피크 0.05초 전부터 샘플링
        cut_start = max(0, start - int(0.05 * sr))
        y_cut = y[cut_start : cut_start + duration]
        
        # 길이 미달 시 제로 패딩
        if len(y_cut) < duration:
            y_cut = np.pad(y_cut, (0, duration - len(y_cut)))
            
        # 논문 규격: 64 mels, hop_length 500
        mel = librosa.feature.melspectrogram(y=y_cut, sr=sr, n_fft=1024, 
                                             hop_length=500, n_mels=64)
        # 개별 키의 선명도를 위해 ref=np.max 사용
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # 64x64 리사이징 및 저장
        final_img = cv2.resize(mel_db, (64, 64))
        plt.imsave(f"{output_dir}/key_{i+1:02d}.png", final_img, cmap='magma')
        print(f"  > [Save] key_{i+1:02d}.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/mel_spectrogram.py <audio_file>")
    else:
        run_pipeline(sys.argv[1])