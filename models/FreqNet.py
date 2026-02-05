import torch
import torch.nn as nn
import torch.fft

# ===== 权重初始化 =====
def _init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ===== FFT模块（实部+虚部） =====
class FFTFeatureExtractor(nn.Module):
    def __init__(self, fs, T, freq_range=(0.5, 5.0)):
        super().__init__()
        freqs = torch.fft.rfftfreq(n=T, d=1/fs)
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        self.register_buffer("fft_mask", mask)
        self.num_freqs = int(mask.sum())
        self.output_dim = 2 * self.num_freqs  # 实部 + 虚部

    def forward(self, x):  # (B, C, T)
        fft = torch.fft.rfft(x, dim=-1)
        real = fft.real[:, :, self.fft_mask]   # (B, C, F)
        imag = fft.imag[:, :, self.fft_mask]
        return torch.cat([real, imag], dim=-1)  # (B, C, 2F)

# ===== 通道卷积融合 =====
class ChannelCombination(nn.Module):
    def __init__(self, C, T, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(C, C, kernel_size=1)
        self.ln1 = nn.LayerNorm([C, T])
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # (B, C, T)
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        return x

# ===== FreqAttention: 跨频率维注意力 =====
class FreqAttention(nn.Module):
    def __init__(self, C, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(C, C // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(C // reduction, C, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # (B, C, F)
        w = self.pool(x)    # (B, C, 1)
        w = self.fc(w)      # (B, C, 1)
        return x * w        # (B, C, F)

# ===== SEBlock: 跨通道注意力 =====
class SEBlock(nn.Module):
    def __init__(self, C, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(C, C // reduction),
            nn.ReLU(),
            nn.Linear(C // reduction, C),
            nn.Sigmoid()
        )

    def forward(self, x):  # (B, C, F)
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w

# ===== 分类头 =====
class MLPHead(nn.Module):
    def __init__(self, input_dim, n_classes, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 6 * n_classes),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(6 * n_classes, n_classes)
        )

    def forward(self, x):  # (B, C×F)
        return self.mlp(x)

# ===== 完整模型 FreqNet =====
class FreqNet(nn.Module):
    def __init__(self, C, T, N, fs=128, freq_range=(0.5, 5.0), dropout=0.3):
        super().__init__()
        self.fft = FFTFeatureExtractor(fs, T, freq_range)
        self.chancomb = ChannelCombination(C, self.fft.output_dim, dropout)
        self.freqatt  = FreqAttention(C)
        self.seblock  = SEBlock(C)
        self.head     = MLPHead(C * self.fft.output_dim, N, dropout)
        self.apply(_init_weights)

    def forward(self, x):  # (B, C, T)
        x = self.fft(x)              # (B, C, 2F)
        x = self.chancomb(x)         # (B, C, 2F)
        x = self.freqatt(x)          # (B, C, 2F)
        x = self.seblock(x)          # (B, C, 2F)
        x = x.flatten(start_dim=1)   # (B, C×2F)
        return self.head(x)          # (B, N)
