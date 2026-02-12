import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from einops import rearrange
import numpy as np
import os

# ==========================================
# 1. CoAtNet 모델 정의 (가변 크기 대응 및 차원 해결)
# ==========================================

def conv_3x3_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MBConv(nn.Module):
    def __init__(self, inp, oup, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        hidden_dim = int(inp * expansion)
        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            SE(inp, hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        return x + self.conv(x)

class Attention(nn.Module):
    def __init__(self, inp, oup, heads=8, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, oup)
        self.attend = nn.Softmax(dim=-1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, inp, oup, heads=8, dim_head=32, downsample=False):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
            current_dim = oup
        else:
            current_dim = inp

        self.attn = Attention(current_dim, oup, heads, dim_head)
        self.ff = nn.Sequential(nn.Linear(oup, oup * 4), nn.GELU(), nn.Linear(oup * 4, oup))
        self.norm1 = nn.LayerNorm(current_dim)
        self.norm2 = nn.LayerNorm(oup)

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x))
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

class CoAtNet(nn.Module):
    def __init__(self, in_channels=3, num_blocks=[2, 2, 3, 5, 2], channels=[64, 96, 192, 384, 768]):
        super().__init__()
        self.s0 = conv_3x3_bn(in_channels, channels[0], stride=2)
        self.s1 = MBConv(channels[0], channels[1], downsample=True)
        self.s2 = self._make_layer('C', channels[1], channels[2], num_blocks[1])
        self.s3 = self._make_layer('T', channels[2], channels[3], num_blocks[2])
        self.s4 = self._make_layer('T', channels[3], channels[4], num_blocks[3])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block_type, inp, oup, depth):
        layers = []
        for i in range(depth):
            down = (i == 0)
            cur_in = inp if down else oup
            if block_type == 'C':
                layers.append(MBConv(cur_in, oup, downsample=down))
            else:
                layers.append(Transformer(cur_in, oup, downsample=down))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.s4(self.s3(self.s2(self.s1(self.s0(x)))))
        return self.pool(x).view(x.size(0), -1)

# ==========================================
# 2. 피처 추출 및 라벨별 폴더 저장 실행부
# ==========================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CoAtNet().to(device)
    model.eval()

    # 경로 설정
    src_path = r'C:\Users\gyulimkim\Desktop\Library\LAB\keyboard_acoustic_side_channel\src\out\1_AULA_F87_Pro'
    node_path = r'C:\Users\gyulimkim\Desktop\Library\LAB\keyboard_acoustic_side_channel\src\node\1_AULA_F87_Pro'
    
    if not os.path.exists(node_path):
        os.makedirs(node_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root=src_path, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"총 {len(dataset)}개의 이미지를 클래스별 폴더에 저장합니다.")

    with torch.no_grad():
        for i, (inputs, label_idx) in enumerate(loader):
            inputs = inputs.to(device)
            features = model(inputs) # 임베딩 추출 (768차원)
            
            # 파일 정보 파싱
            img_path, _ = dataset.samples[i]
            img_name = os.path.basename(img_path).split('.')[0]
            class_name = dataset.classes[label_idx] # 예: 'a', 'b', '1'
            
            # [라벨 폴더 생성 로직]
            class_node_dir = os.path.join(node_path, class_name)
            if not os.path.exists(class_node_dir):
                os.makedirs(class_node_dir)
            
            # 저장 (이진 파일 .npy)
            save_path = os.path.join(class_node_dir, f"{img_name}.npy")
            np.save(save_path, features.cpu().numpy().flatten())
            
            if (i+1) % 100 == 0:
                print(f"[{i+1}/{len(dataset)}] '{class_name}' 폴더 처리 완료...")

    print(f"성공! 피처 임베딩이 {node_path}에 클래스별로 저장되었습니다.")