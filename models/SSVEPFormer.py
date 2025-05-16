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
    def __init__(self, C, F, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(2*C, 2*C, kernel_size=1, padding=0)
        self.ln1   = nn.LayerNorm([2*C, F])
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)

    def forward(self, X_complex):
        # X_complex: (B, C, F), dtype=complex
        real = X_complex.real
        imag = X_complex.imag
        x = torch.cat([real, imag], dim=1)   # -> (B, 2*C, F)
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        return x  # (B, 2*C, F)

class SSVEPFormerEncoderBlock(nn.Module):
    def __init__(self, C, F, dropout=0.5):
        super().__init__()
        D = 2*C
        # ---- CNN Module ----
        self.ln_c1 = nn.LayerNorm([D, F])
        self.conv   = nn.Conv1d(D, D, kernel_size=5, padding=2)
        self.ln_c2 = nn.LayerNorm([D, F])
        self.act_c = nn.GELU()
        self.drop_c= nn.Dropout(dropout)
        # ---- Channel MLP Module ----
        self.ln_m1 = nn.LayerNorm([D, F])
        self.lin    = nn.Linear(F, F)
        self.act_m = nn.GELU()
        self.drop_m= nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, 2*C, F)
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

        return x  # (B, 2*C, F)

class MLPHead(nn.Module):
    def __init__(self, C, F, N, dropout=0.5):
        super().__init__()
        hidden = 6 * N
        self.drop1 = nn.Dropout(dropout)
        self.lin1  = nn.Linear(2*C*F, hidden)
        self.ln    = nn.LayerNorm(hidden)
        self.act   = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.lin2  = nn.Linear(hidden, N)

    def forward(self, x):
        # x: (B, 2*C, F)
        b, c2, f = x.shape
        x = x.view(b, c2*f)
        x = self.drop1(x)
        x = self.lin1(x)   # -> (B, 6*N)
        x = self.ln(x)
        x = self.act(x)
        x = self.drop2(x)
        x = self.lin2(x)   # -> (B, N)
        return x

class SSVEPFormer(nn.Module):
    def __init__(self, C, T, N, fs, freq_range=None, dropout=0.5):
        """
        C: number of channels
        T: time-domain length
        N: number of classes
        fs: sampling rate (Hz)
        freq_range: tuple (f_min, f_max) in Hz to slice spectrum; if None, use all
        """
        super().__init__()
        # full FFT bins
        F_full = T//2 + 1

        # determine which frequency bins to keep
        if freq_range is not None:
            fmin, fmax = freq_range
            freqs = np.fft.rfftfreq(T, d=1/fs)
            mask = (freqs >= fmin) & (freqs <= fmax)
            idx = np.where(mask)[0]
            self.register_buffer('freq_idx', torch.LongTensor(idx))
            F = len(idx)
        else:
            self.freq_idx = None
            F = F_full

        self.fs = fs
        self.chancomb = ChannelCombination(C, F, dropout)
        self.encoder = nn.Sequential(
            SSVEPFormerEncoderBlock(C, F, dropout),
            SSVEPFormerEncoderBlock(C, F, dropout),
        )
        self.head = MLPHead(C, F, N, dropout)

        # weight init as per paper
        self.apply(_init_weights)

    def forward(self, x):
        # x: (B, C, T)
        # compute complex spectrum
        Xf = torch.fft.rfft(x, dim=-1)  # (B, C, F_full)
        # optional slice
        if self.freq_idx is not None:
            Xf = Xf[..., self.freq_idx]   # -> (B, C, F)
        z0 = self.chancomb(Xf)           # (B, 2*C, F)
        z1 = self.encoder(z0)            # (B, 2*C, F)
        out = self.head(z1)              # (B, N)
        return out

# ======= 简单测试 =======
if __name__ == "__main__":
    batch, C, T, N = 8, 9, 256, 6
    fs = 128
    # 不指定 freq_range 时用全部频点
    model_all = SSVEPFormer(C=C, T=T, N=N, fs=fs, freq_range=None)
    x = torch.randn(batch, C, T)
    y = model_all(x)
    print("Output shape (all freqs):", y.shape)

    # 指定 8–64 Hz
    model_8_64 = SSVEPFormer(C=C, T=T, N=N, fs=fs, freq_range=(8, 64))
    y2 = model_8_64(x)
    print("Output shape (8–64Hz):", y2.shape)
