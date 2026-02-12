import os
import numpy as np
import torch
from torch_geometric.data import Data

def prepare_keyboard_dataset(base_path):
    all_data = []
    folder_names = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
    label_map = {name: i for i, name in enumerate(folder_names)}
    
    num_nodes = 46 # 고정된 노드 수
    node_identities = torch.eye(num_nodes) 

    print(f"데이터 변환 중... (클래스: {len(folder_names)}개)")

    for folder in folder_names:
        folder_path = os.path.join(base_path, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        
        for file in files:
            feature = np.load(os.path.join(folder_path, file))
            x_raw = torch.from_numpy(feature).float().view(1, 768)
            
            # [46, 768] 복제 + [46, 46] 식별자 = [46, 814]
            node_features = torch.cat([x_raw.repeat(num_nodes, 1), node_identities], dim=-1)
            label = torch.tensor([label_map[folder]], dtype=torch.long)
            
            # 전결합 에지
            r = torch.arange(num_nodes)
            edge_index = torch.stack([torch.repeat_interleave(r, num_nodes), torch.tile(r, (num_nodes,))])
            
            all_data.append(Data(x=node_features, edge_index=edge_index, y=label))
            
    return all_data, label_map

if __name__ == "__main__":
    BASE_PATH = "/content/drive/MyDrive/src/node/1_AULA_F87_Pro"
    SAVE_PATH = "/content/drive/MyDrive/src/processed_dataset.pt"
    
    dataset, label_dict = prepare_keyboard_dataset(BASE_PATH)
    torch.save({'dataset': dataset, 'label_dict': label_dict}, SAVE_PATH)
    print(f"성공적으로 저장되었습니다: {SAVE_PATH}")