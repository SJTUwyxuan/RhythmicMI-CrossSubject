# ======== models/FreqMLP_SAD.py ========

import numpy as np
import torch
import torch.nn as nn

def _init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ChannelCombination(nn.Module):
    def __init__(self, C, F, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(2 * C, 2 * C, kernel_size=1)
        self.ln1   = nn.LayerNorm([2 * C, F])
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)

    def forward(self, X_complex):
        real = X_complex.real
        imag = X_complex.imag
        x = torch.cat([real, imag], dim=1)  # (B,2C,F)
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act(x)
        return self.drop(x)

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

class SubjectDiscriminator(nn.Module):
    def __init__(self, in_dim, n_subjects):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_subjects)
        )

    def forward(self, x):
        return self.net(x)

class FreqMLP_SAD(nn.Module):
    def __init__(self, C, T, N, fs, freq_range=None, dropout=0.5, n_subjects=56):
        super().__init__()
        F_full = T // 2 + 1
        if freq_range is not None:
            fmin, fmax = freq_range
            freqs = np.fft.rfftfreq(T, d=1 / fs)
            mask  = (freqs >= fmin) & (freqs <= fmax)
            idx   = np.where(mask)[0]
            self.register_buffer("freq_idx", torch.LongTensor(idx))
            F = len(idx)
        else:
            self.freq_idx = None
            F = F_full

        self.C = C
        self.F = F
        self.feature       = ChannelCombination(C, F, dropout)
        self.classifier    = MLPHead(C, F, N, dropout)
        self.discriminator = SubjectDiscriminator(2 * C * F, n_subjects)
        self.apply(_init_weights)

    def forward(self, x):
        Xf = torch.fft.rfft(x, dim=-1)
        if self.freq_idx is not None:
            Xf = Xf[..., self.freq_idx]
        feat      = self.feature(Xf)
        flat_feat = feat.view(feat.size(0), -1)
        y_pred    = self.classifier(feat)
        s_pred    = self.discriminator(flat_feat)
        return y_pred, s_pred

    def extract_features(self, x):
        Xf = torch.fft.rfft(x, dim=-1)
        if self.freq_idx is not None:
            Xf = Xf[..., self.freq_idx]
        feat = self.feature(Xf)
        return feat.view(feat.size(0), -1)
