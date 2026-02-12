import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

# 1. 기본적인 3x3 Conv + BN + GELU 블록
def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 2 if downsample else 1
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

# 2. SE(Squeeze-and-Excitation) 블록: 채널 간의 중요도를 재보정
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

# 3. MBConv 블록: CNN 스테이지(S0, S1, S2)에서 사용
class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
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

# 4. Attention 블록: Transformer 스테이지(S3, S4)에서 사용
class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.ih, self.iw = image_size
        self.heads = heads
        self.scale = dim_head ** -0.5

        # Relative Positional Bias (논문에서 강조된 부분)
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))
        
        coords = torch.stack(torch.meshgrid(torch.arange(self.ih), torch.arange(self.iw), indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.ih - 1
        relative_coords[:, :, 1] += self.iw - 1
        relative_coords[:, :, 0] *= 2 * self.iw - 1
        relative_index = relative_coords.sum(-1)
        self.register_buffer("relative_index", relative_index)

        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, oup)
        self.attend = nn.Softmax(dim=-1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        bias = self.relative_bias_table[self.relative_index.view(-1)].view(
            self.ih * self.iw, self.ih * self.iw, -1).permute(2, 0, 1).contiguous()
        dots = dots + bias.unsqueeze(0)

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 5. Transformer 블록 구성
class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        ih, iw = image_size
        self.downsample = downsample
        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        
        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = nn.Sequential(
            nn.Linear(oup, oup * 4),
            nn.GELU(),
            nn.Linear(oup * 4, oup)
        )
        self.norm1 = nn.LayerNorm(inp)
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

# 6. 전체 CoAtNet 구조 정의
class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=36):
        super().__init__()
        ih, iw = image_size
        self.s0 = conv_3x3_bn(in_channels, channels[0], image_size, downsample=True)
        self.s1 = MBConv(channels[0], channels[1], (ih//2, iw//2), downsample=True)
        
        self.s2 = self._make_layer('C', channels[1], channels[2], num_blocks[1], (ih//4, iw//4))
        self.s3 = self._make_layer('T', channels[2], channels[3], num_blocks[2], (ih//8, iw//8))
        self.s4 = self._make_layer('T', channels[3], channels[4], num_blocks[3], (ih//16, iw//16))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[4], num_classes)

    def _make_layer(self, block_type, inp, oup, depth, image_size):
        layers = []
        for i in range(depth):
            downsample = True if i == 0 else False
            if block_type == 'C':
                layers.append(MBConv(inp if i == 0 else oup, oup, image_size, downsample))
            else:
                layers.append(Transformer(inp if i == 0 else oup, oup, image_size, downsample=downsample))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.s4(self.s3(self.s2(self.s1(self.s0(x)))))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# 7. CoAtNet-0 사양 생성 함수
def coatnet_0(num_classes=36):
    num_blocks = [2, 2, 3, 5, 2]            # 논문 기준 Stage별 블록 수
    channels = [64, 96, 192, 384, 768]      # 논문 기준 Stage별 채널 수
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes)

# 테스트 코드
if __name__ == "__main__":
    img = torch.randn(1, 3, 224, 224)
    model = coatnet_0(num_classes=36)
    out = model(img)
    print(out.shape) # torch.Size([1, 36])