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

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):  # x: (B, T, hidden)
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        pooled = torch.sum(x * weights, dim=1)        # (B, hidden)
        return pooled

# class LSTMEncoderSimple(nn.Module):
#     def __init__(self, C, hidden_size=64, num_layers=1, dropout=0.5):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_size=C,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0
#         )

#     def forward(self, x):
#         # x: (B, C, T) → (B, T, C)
#         x = x.permute(0, 2, 1)
#         out, _ = self.lstm(x)
#         return out.mean(dim=1)  # (B, hidden_size)

class LSTMEncoderWithAttention(nn.Module):
    def __init__(self, C, hidden_size=128, num_layers=1, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=C,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.pooling = AttentionPooling(hidden_size)

    def forward(self, x):
        # x: (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)         # (B, T, hidden)
        pooled = self.pooling(out)   # (B, hidden)
        return pooled


class Classifier(nn.Module):
    def __init__(self, hidden_size, N, dropout=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, N)
        )

    def forward(self, x):
        return self.classifier(x)  # (B, N)

class SPTNet(nn.Module):
    def __init__(self, C, T, N, fs, freq_range=None, hidden_size=128, dropout=0.5):
        super().__init__()
        self.fs = fs
        self.chancomb  = ChannelCombination(C, T, dropout)
        self.encoder   = LSTMEncoderWithAttention(C, hidden_size, num_layers=1, dropout=dropout)
        self.classifier = Classifier(hidden_size, N, dropout)

        self.apply(_init_weights)

    def forward(self, x):
        # x: (B, C, T)
        z0 = self.chancomb(x)     # (B, C, T)
        z1 = self.encoder(z0)     # (B, hidden_size)
        out = self.classifier(z1) # (B, N)
        return out

# ======= 简单测试 =======
if __name__ == "__main__":
    batch, C, T, N = 8, 9, 256, 6
    fs = 128
    model = SPTNet(C=C, T=T, N=N, fs=fs)
    x = torch.randn(batch, C, T)
    y = model(x)
    print("Output shape:", y.shape)
