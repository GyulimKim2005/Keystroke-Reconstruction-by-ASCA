import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from PIL import Image
import os

# 1. 특징 추출용 래퍼 클래스 정의
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model
        # CoAtNet의 마지막 분류 레이어(fc/head)를 제거하고 직전의 특징만 반환
        self.feature_layer = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.feature_layer(x)
        return torch.flatten(x, 1) # 1차원 벡터로 변환

def visualize_similarity_matrix(image_paths, labels, model):
    """
    이미지 리스트를 넣어 n x n 유사도 행렬을 그립니다.
    """
    model.eval()
    extractor = FeatureExtractor(model)
    
    # 이미지 전처리 (64x64 리사이징 및 정규화)
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    
    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0)
            # 모델로부터 특징 벡터(Feature Vector) 추출
            feat = extractor(img_tensor)
            features.append(feat.numpy().flatten())

    # 2. 코사인 유사도 계산 (n x n matrix)
    features = np.array(features)
    sim_matrix = cosine_similarity(features)

    # 3. 히트맵 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap='YlGnBu',
                xticklabels=labels, yticklabels=labels)
    plt.title("Keystroke Spectrogram Feature Similarity Matrix")
    plt.show()

# 사용 예시 (학습된 모델이 'model' 변수에 저장되어 있다고 가정)
# image_list = ["path/to/A.png", "path/to/B.png", "path/to/Shift.png", ...]
# label_list = ["Key_A", "Key_B", "Key_Shift", ...]
# visualize_similarity_matrix(image_list, label_list, model)