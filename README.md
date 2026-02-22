Keystroke Reconstruction by ASCA - 파일 설명

아래는 이 폴더에 있는 주요 파일/디렉터리의 역할 요약입니다.

모델 정의
- `CoAtNet.py`: CoAtNet 모델 구현(Conv/MBConv/Attention/Transformer 블록).
- `GaAN.py`: 키보드 그래프 생성 유틸과 GaAN 기반 분류 모델 정의(예시 실행 포함).
- `GaAN_node.py`: 노드 분류용 간단한 GATv2 기반 모델 정의.
- `feature_extraction.py`: CoAtNet 기반 특징 추출 파이프라인/모델 정의(가변 크기 대응).

파일
- `dataset_stroke.py`: 키 입력 시퀀스 그래프 데이터셋 생성 스크립트(npz의 `x`, `iki` 사용).
- `train_node.py`: `dataset_stroke.py`로 만든 데이터셋을 불러 GaAN 노드 분류 모델을 학습.
- `coatnet_classifier.pth`: 학습된 CoAtNet 분류기 가중치 파일.
- `mel_spec_for_test_data.ipynb`: 테스트 데이터용 멜-스펙트로그램 생성, CoAtNet에서 feature extraction, npz파일 생성
- `mel_spectogram_generator.ipynb`: CoAtNet학습 데이터용 멜-스펙트로그램 생성 
- `train_CoAtNet.ipynb`: CoAtNet train

- `npz_sequences/*.npz`: 한 타이핑 시퀀스(문장/단어) 단위 데이터. 내부에 `x`(shape `[N, 768]`, keystroke 특징), `iki`(shape `[N-1]`, 인접 keystroke 간 시간 간격)를 포함.


