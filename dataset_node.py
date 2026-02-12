import os
import numpy as np
import torch
from torch_geometric.data import Data

def prepare_keyboard_dataset(base_path):
    all_data = []
    # 폴더명 리스트를 정렬하여 일관된 라벨링 부여 (예: A=0, B=1...)
    folder_names = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
    label_map = {name: i for i, name in enumerate(folder_names)}
    
    print(f"총 {len(folder_names)}개의 클래스(키)를 발견했습니다.")

    for folder in folder_names:
        folder_path = os.path.join(base_path, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            # 1. 피처 로드 (768,)
            feature = np.load(file_path)
            x = torch.from_numpy(feature).float().view(1, 768)
            
            # 2. 46개 노드 구조로 확장 (현재 신호를 모든 노드에 뿌려줌)
            # 실제 연구에서는 특정 위치에만 피처를 넣는 방식으로 고도화 가능
            node_features = x.repeat(46, 1) 
            
            # 3. 라벨 설정
            label = torch.tensor([label_map[folder]], dtype=torch.long)
            
            # 4. 전결합 에지 생성 (46x46)
            num_nodes = 46
            edge_index = torch.stack([
                torch.repeat_interleave(torch.arange(num_nodes), num_nodes),
                torch.tile(torch.arange(num_nodes), (num_nodes,))
            ])
            
            # PyG 데이터 객체 생성
            data = Data(x=node_features, edge_index=edge_index, y=label)
            all_data.append(data)
            
    print(f"총 {len(all_data)}개의 데이터 샘플을 로드했습니다.")
    return all_data, label_map

# 실행 예시
base_path = r"/content/drive/MyDrive/src/node/1_AULA_F87_Pro"
dataset, label_dict = prepare_keyboard_dataset(base_path)