# === 最小改动地引入 DANN-style 域对抗结构 ===
# 本方案保持训练主流程不变，仅对模型和训练循环稍作调整

# ======= 新建文件: models/FreqMLP_DANN.py =======
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def _init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ---------- GRL ----------
class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReverseLayer.apply(x, lambda_)

# ---------- 模型定义 ----------
class ChannelCombination(nn.Module):
    def __init__(self, C, F, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(2 * C, 2 * C, kernel_size=1)
        self.ln1 = nn.LayerNorm([2 * C, F])
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, X_complex):
        real = X_complex.real
        imag = X_complex.imag
        x = torch.cat([real, imag], dim=1)
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class MLPHead(nn.Module):
    def __init__(self, C, F, N, dropout=0.5):
        super().__init__()
        hidden = 6 * N
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * C * F, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, N),
        )

    def forward(self, x):
        b, c, f = x.shape
        x = x.view(b, c * f)
        return self.net(x)

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, num_domains):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_domains)
        )

    def forward(self, x):
        return self.discriminator(x)

class FreqMLP_DANN(nn.Module):
    def __init__(self, C, T, N, fs, freq_range=None, dropout=0.5, num_domains=20):
        super().__init__()
        F_full = T // 2 + 1
        if freq_range is not None:
            fmin, fmax = freq_range
            freqs = np.fft.rfftfreq(T, d=1 / fs)
            mask = (freqs >= fmin) & (freqs <= fmax)
            idx = np.where(mask)[0]
            self.register_buffer("freq_idx", torch.LongTensor(idx))
            F = len(idx)
        else:
            self.freq_idx = None
            F = F_full

        self.C = C
        self.F = F

        self.chancomb = ChannelCombination(C, F, dropout)
        self.head = MLPHead(C, F, N, dropout)
        self.apply(_init_weights)
        self.domain_disc = DomainDiscriminator(input_dim=2 * C * F, num_domains=num_domains)

    def forward(self, x, lambda_grl=1.0):
        Xf = torch.fft.rfft(x, dim=-1)
        if self.freq_idx is not None:
            Xf = Xf[..., self.freq_idx]
        x = self.chancomb(Xf)             # (B, 2C, F)
        flat = x.view(x.size(0), -1)      # domain classifier 用

        class_output = self.head(x)       # task 分类输出
        reversed_feat = grad_reverse(flat, lambda_grl)
        domain_output = self.domain_disc(reversed_feat)
        return class_output, domain_output
