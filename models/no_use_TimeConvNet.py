import torch
import torch.nn as nn

def _init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

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

class TemporalConvGLU(nn.Module):
    def __init__(self, C, kernel_size=7, dropout=0.3):
        super().__init__()
        self.depthwise = nn.Conv1d(C, 2*C, kernel_size, padding=kernel_size//2, groups=C)
        self.pointwise = nn.Conv1d(C, C, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # (B, C, T)
        proj = self.pointwise(x)
        y = self.depthwise(x)
        A, B = y.chunk(2, dim=1)
        y = A * torch.sigmoid(B)
        return self.drop(y + proj)

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

    def forward(self, x):  # (B, C, T)
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w

class MLPHead(nn.Module):
    def __init__(self, C, T, N, dropout=0.3):
        super().__init__()
        hidden = 6 * N
        self.net = nn.Sequential(
            nn.LayerNorm(C * T),
            nn.Linear(C * T, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, N)
        )

    def forward(self, x):  # (B, C, T)
        x = x.flatten(start_dim=1)  # (B, C*T)
        return self.net(x)

class TimeConvNet(nn.Module):
    def __init__(self, C, T, N, fs=128, freq_range=None, dropout=0.3):
        super().__init__()
        self.chancomb = ChannelCombination(C, T, dropout)
        self.temporal = TemporalConvGLU(C, dropout=dropout)
        self.se       = SEBlock(C)
        self.head     = MLPHead(C, T, N, dropout)
        self.apply(_init_weights)

    def forward(self, x):  # x: (B, C, T)
        x = self.chancomb(x)  # (B, C, T)
        x = self.temporal(x)  # (B, C, T)
        x = self.se(x)        # (B, C, T)
        out = self.head(x)    # (B, N)
        return out
