# ======== TimeMLP.py ========
import torch
import torch.nn as nn


def _init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ChannelCombination(nn.Module):
    def __init__(self, C, T, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(C, C, kernel_size=1)
        self.ln1 = nn.LayerNorm([C, T])
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, C, T, N, dropout=0.5):
        super().__init__()
        hidden = 6 * N
        self.net = nn.Sequential(
            nn.Linear(C * T, hidden),
            nn.LayerNorm(hidden),
            # nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, N),
        )

    def forward(self, x):
        b, c, t = x.shape
        x = x.view(b, c * t)
        return self.net(x)


class TimeMLP(nn.Module):
    def __init__(self, C, T, N, fs, freq_range=None, dropout=0.5):
        super().__init__()
        self.chancomb = ChannelCombination(C, T, dropout)
        self.head = MLPHead(C, T, N, dropout)
        self.apply(_init_weights)

    def forward(self, x):
        x = self.chancomb(x)
        out = self.head(x)
        return out
