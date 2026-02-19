import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class SimpleKeyboardGaAN(nn.Module):
    def __init__(self, in_channels=768, hidden_channels=128, num_classes=46, edge_dim=1):
        super(SimpleKeyboardGaAN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.edge_dim = edge_dim

        # edge_attr 사용을 위해 GATv2Conv 사용 (edge_dim 지정)
        self.conv1 = GATv2Conv(self.in_channels, self.hidden_channels, heads=4, concat=True, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(self.hidden_channels * 4, self.hidden_channels, heads=1, concat=False, edge_dim=edge_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.num_classes)
        )

    def forward(self, x, edge_index, edge_attr):
        # x shape: [Total_Nodes, 768]
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        # 노드별 분류
        return self.fc(x)
