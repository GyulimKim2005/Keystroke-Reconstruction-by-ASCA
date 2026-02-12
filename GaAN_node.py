import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleKeyboardGaAN(nn.Module):
    def __init__(self, in_channels=768, out_channels=64, heads=8):
        super(SimpleKeyboardGaAN, self).__init__()
        self.heads = heads
        self.out_channels = out_channels
        
        # 1. 노드 정체성 주입: 46개 노드 각각에 고유 ID(a, b, 1...) 부여
        # 이 이름표가 있어야 모델이 'A' 위치 소리를 'A'로 인식합니다.
        self.node_identity = nn.Embedding(46, in_channels)
        
        # 2. GaAN의 핵심: Multi-head Attention
        self.q_lin = nn.Linear(in_channels, out_channels * heads)
        self.k_lin = nn.Linear(in_channels, out_channels * heads)
        self.v_lin = nn.Linear(in_channels, out_channels * heads)
        
        # 3. 게이트 유닛: 기종별/주파수별 노이즈 필터
        self.gate_lin = nn.Linear(in_channels * 2, heads)
        
        self.classifier = nn.Linear(out_channels * heads, 46)

    def forward(self, x, edge_index=None):
        # 입력 형태 보정 [Batch * 46, 768] -> [Batch, 46, 768]
        if x.dim() == 2: x = x.view(-1, 46, 768)
        batch_size, num_nodes, _ = x.size()
        
        # [핵심] 소리에 노드 이름표(Identity)를 더함
        node_ids = torch.arange(num_nodes, device=x.device)
        x = x + self.node_identity(node_ids) 
        
        # Multi-head Attention 연산
        q = self.q_lin(x).view(batch_size, num_nodes, self.heads, self.out_channels).transpose(1, 2)
        k = self.k_lin(x).view(batch_size, num_nodes, self.heads, self.out_channels).transpose(1, 2)
        v = self.v_lin(x).view(batch_size, num_nodes, self.heads, self.out_channels).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.out_channels ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        # 게이트 유닛으로 유효 헤드 선별
        global_feat = x.mean(dim=1, keepdim=True).expand(-1, num_nodes, -1)
        gate = torch.sigmoid(self.gate_lin(torch.cat([x, global_feat], dim=-1)))
        
        out = torch.matmul(attn, v).transpose(1, 2) # [B, N, H, D]
        out = out * gate.unsqueeze(-1) # 게이트 적용
        
        # 최종 분류
        out = out.reshape(batch_size * num_nodes, -1)
        return self.classifier(out)