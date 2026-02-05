import numpy as np
import torch
import torch.nn as nn

class FFTLinearClassifier(nn.Module):
    def __init__(self, C, T, N, fs, freq_range=None, dropout=0.5):
        super().__init__()
        F_full = T // 2 + 1

        if freq_range is not None:
            fmin, fmax = freq_range
            freqs = np.fft.rfftfreq(T, d=1 / fs)
            mask = (freqs >= fmin) & (freqs <= fmax)
            idx = np.where(mask)[0]
            self.register_buffer('freq_idx', torch.LongTensor(idx))
            F = len(idx)
        else:
            self.freq_idx = None
            F = F_full

        self.flatten_dim = 2 * C * F
        hidden_dim = 6 * N

        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, N)
        )

    def forward(self, x):
        Xf = torch.fft.rfft(x, dim=-1)  # (B, C, F_full)
        if self.freq_idx is not None:
            Xf = Xf[..., self.freq_idx]  # (B, C, F)
        real = Xf.real
        imag = Xf.imag
        x = torch.cat([real, imag], dim=1)  # (B, 2C, F)
        x = x.view(x.size(0), -1)
        return self.net(x)

# 测试代码
if __name__ == "__main__":
    batch, C, T, N = 8, 9, 256, 6
    fs = 128
    model = FFTLinearClassifier(C=C, T=T, N=N, fs=fs, freq_range=(8, 64))
    x = torch.randn(batch, C, T)
    y = model(x)
    print("Output shape:", y.shape)
