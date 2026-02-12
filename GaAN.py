import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import grid # 실제 위치 기반일 때 유용하지만 여기선 fc 사용

# --- 1. 전결합 에지 및 IKT 데이터 생성 함수 ---
def create_keyboard_graph(node_features, ikt_matrix):
    """
    node_features: [46, 768] (CoAtNet 추출 피처)
    ikt_matrix: [46, 46] (각 키 사이의 시간 간격 데이터)
    """
    num_nodes = node_features.size(0)
    
    # 46x46 모든 연결 생성 (Self-loop 포함)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(num_nodes), num_nodes),
        torch.tile(torch.arange(num_nodes), (num_nodes,))
    ])
    
    # 에지 속성에 IKT 값 할당 [2070, 1]
    edge_attr = ikt_matrix.view(-1, 1)
    
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# --- 2. GaAN 모델 정의 (주파수 대역 Gating 강조) ---
class KeyboardGaAN(nn.Module):
    def __init__(self, in_channels=768, hidden_dim=64, heads=8, num_classes=46):
        super(KeyboardGaAN, self).__init__()
        
        # 주파수 대역(768차원)을 8개 헤드로 나누어 분석하는 GaAN 레이어
        self.gaan1 = GaANLayer(in_channels, hidden_dim, heads=heads)
        self.gaan2 = GaANLayer(hidden_dim * heads, hidden_dim, heads=heads)
        
        # 최종 Keystroke 복원 레이어
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim * heads, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes) # 46개 클래스 분류
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 첫 번째 레이어 (주파수 특징 추출 및 게이팅)
        x = torch.relu(self.gaan1(x, edge_index, edge_attr))
        
        # 두 번째 레이어 (복합 관계 파악)
        x = torch.relu(self.gaan2(x, edge_index, edge_attr))
        
        # 최종 노드별 분류 (이 노드가 실제로 눌린 키일 확률)
        logits = self.reconstruction_head(x)
        return logits

# --- 3. 실행 예시 ---
# 1_001.npy 데이터를 46개 노드용으로 가공했다고 가정
sample_features = torch.randn(46, 768) 
sample_ikt = torch.rand(46, 46) # 실제 데이터셋의 IKT 값 대입 필요

graph_data = create_keyboard_graph(sample_features, sample_ikt)
model = KeyboardGaAN()
output = model(graph_data)

print(f"출력 사이즈 (노드수, 클래스수): {output.shape}") # [46, 46]