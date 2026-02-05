import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ChannelCombination(nn.Module):
    def __init__(self, C, T, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(C, C, kernel_size=1, padding=0)
        self.ln1   = nn.LayerNorm([C, T])
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):

        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        return x  # (B, C, T)

class TimeFormerEncoderBlock(nn.Module):
    def __init__(self, C, T, dropout=0.5):
        super().__init__()
        D = C
        # ---- CNN Module ----
        self.ln_c1 = nn.LayerNorm([D, T])
        self.conv   = nn.Conv1d(D, D, kernel_size=5, padding='same')
        self.ln_c2 = nn.LayerNorm([D, T])
        self.act_c = nn.GELU()
        self.drop_c= nn.Dropout(dropout)
        # ---- Channel MLP Module ----
        self.ln_m1 = nn.LayerNorm([D, T])
        self.lin    = nn.Linear(T, T)
        self.act_m = nn.GELU()
        self.drop_m= nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        # CNN branch + residual
        y = self.ln_c1(x)
        y = self.conv(y)
        y = self.ln_c2(y)
        y = self.act_c(y)
        y = self.drop_c(y)
        x = x + y  # Residual1

        # Channel-MLP branch + residual
        z = self.ln_m1(x)
        b, d, f = z.shape
        z = z.view(b*d, f)
        z = self.lin(z)
        z = self.act_m(z)
        z = self.drop_m(z)
        z = z.view(b, d, f)
        x = x + z  # Residual2

        return x  # (B, C, T)

class MLPHead(nn.Module):
    def __init__(self, C, T, N, dropout=0.5):
        super().__init__()
        hidden = 6 * N
        self.drop1 = nn.Dropout(dropout)
        self.lin1  = nn.Linear(C*T, hidden)
        self.ln    = nn.LayerNorm(hidden)
        self.act   = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.lin2  = nn.Linear(hidden, N)

    def forward(self, x):
        # x: (B, C, T)
        b, c, t = x.shape
        x = x.view(b, c*t)
        x = self.drop1(x)
        x = self.lin1(x)   # -> (B, 6*N)
        x = self.ln(x)
        x = self.act(x)
        x = self.drop2(x)
        x = self.lin2(x)   # -> (B, N)
        return x

class TimeFormer(nn.Module):
    def __init__(self, C, T, N, fs, freq_range=None, dropout=0.5):
        """
        C: number of channels
        T: time-domain length
        N: number of classes
        fs: sampling rate (Hz)
        freq_range: tuple (f_min, f_max) in Hz to slice spectrum; if None, use all
        """
        super().__init__()

        self.fs = fs
        self.chancomb = ChannelCombination(C, T, dropout)
        self.encoder = nn.Sequential(
            TimeFormerEncoderBlock(C, T, dropout),
            TimeFormerEncoderBlock(C, T, dropout),
        )
        self.head = MLPHead(C, T, N, dropout)

        # weight init as per paper
        self.apply(_init_weights)

    def forward(self, x):
        # x: (B, C, T)
        z0 = self.chancomb(x)           # (B, C, T)
        z1 = self.encoder(z0)            # (B, C, T)
        out = self.head(z1)              # (B, N)
        return out

# ======= 简单测试 =======
if __name__ == "__main__":
    batch, C, T, N = 8, 9, 256, 6
    fs = 128
    # 不指定 freq_range 时用全部频点
    model_all = TimeFormer(C=C, T=T, N=N, fs=fs, freq_range=None)
    x = torch.randn(batch, C, T)
    y = model_all(x)
    print("Output shape (all freqs):", y.shape)

    # 指定 8–64 Hz
    model_8_64 = TimeFormer(C=C, T=T, N=N, fs=fs, freq_range=(8, 64))
    y2 = model_8_64(x)
    print("Output shape (8–64Hz):", y2.shape)
