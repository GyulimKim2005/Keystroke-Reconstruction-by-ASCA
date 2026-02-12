import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class SimpleKeyboardGaAN(nn.Module):
    def __init__(self):
        super(SimpleKeyboardGaAN, self).__init__()
        # 입력 차원: 768(신호) + 46(노드ID) = 814
        self.in_channels = 814 
        self.hidden_channels = 128
        self.num_classes = 46

        self.conv1 = GATConv(self.in_channels, self.hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(self.hidden_channels * 4, self.hidden_channels, heads=1, concat=False)
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.num_classes)
        )

    def forward(self, x, edge_index, batch=None):
        # x shape: [Batch_Size * 46, 814]
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # 그래프 분류를 위해 노드들의 특징을 하나로 합침 (Global Pooling)
        # batch 인자가 없으면 단일 그래프로 간주
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        x = global_mean_pool(x, batch) # [Batch_Size, hidden_channels]
        return self.fc(x)