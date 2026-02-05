# ======== FreqMLP.py ========
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
        self.conv1 = nn.Conv1d(C, C, kernel_size=1)
        self.ln1 = nn.LayerNorm([C, F])
        # self.ln1 = nn.BatchNorm1d([2 * C])
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, X_complex):
        real = X_complex.real
        imag = X_complex.imag
        x = real ** 2 + imag ** 2
        # x = imag
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
            nn.Linear(C * F, hidden),
            nn.LayerNorm(hidden),
            # nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, N),
        )

    def forward(self, x):
        b, c, f = x.shape
        x = x.view(b, c * f)
        return self.net(x)


# class MLPHead(nn.Module):
#     def __init__(self, C, F, N, dropout=0.5):
#         super().__init__()
#         hidden = 6 * N
#         self.net = nn.Sequential(

#             nn.Dropout(dropout),
#             nn.Linear(2 * C * F, 4 * C * F),
#             nn.LayerNorm(4 * C * F),
#             nn.GELU(),

#             nn.Dropout(dropout),
#             nn.Linear(4 * C * F, 2*hidden),
#             nn.LayerNorm(2*hidden),
#             nn.GELU(),

#             nn.Dropout(dropout),
#             nn.Linear(2*hidden, hidden),
#             nn.LayerNorm(hidden),
#             nn.GELU(),

#             nn.Dropout(dropout),
#             nn.Linear(hidden, N),
#         )

#     def forward(self, x):
#         b, c, f = x.shape
#         x = x.view(b, c * f)
#         return self.net(x)


class FreqAmpMLP(nn.Module):
    def __init__(self, C, T, N, fs, freq_range=None, dropout=0.5):
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

        self.chancomb = ChannelCombination(C, F, dropout)
        self.head = MLPHead(C, F, N, dropout)
        self.apply(_init_weights)

    def forward(self, x):
        Xf = torch.fft.rfft(x, dim=-1)
        if self.freq_idx is not None:
            Xf = Xf[..., self.freq_idx]
        x = self.chancomb(Xf)
        out = self.head(x)
        return out