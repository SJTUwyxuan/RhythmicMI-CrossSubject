import torch
import torch.nn as nn
import torch.fft

# def _init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.Conv1d):
#         nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        return x  # (B, C, T)

class FFTFeatureExtractor(nn.Module):
    def __init__(self, fs, T, freq_range=(0.5, 5.0)):
        super().__init__()
        freqs = torch.fft.rfftfreq(n=T, d=1/fs)  # shape (F,)
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        self.register_buffer("fft_mask", mask)
        self.num_freqs = int(mask.sum().item())  # 实部数量
        self.output_dim = 2 * self.num_freqs     # 实部 + 虚部

    def forward(self, x):
        # x: (B, C, T)
        fft = torch.fft.rfft(x, dim=-1)                # (B, C, F)
        real = fft.real[:, :, self.fft_mask]           # (B, C, F_selected)
        imag = fft.imag[:, :, self.fft_mask]           # (B, C, F_selected)
        return torch.cat([real, imag], dim=-1)         # (B, C, 2 × F_selected)

class MLPHead(nn.Module):
    def __init__(self, input_dim, n_classes, dropout=0.3):
        super().__init__()
        hidden = 4 * n_classes
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x):  # x: (B, C×F)
        return self.mlp(x)

class RhythmMLP(nn.Module):
    def __init__(self, C, T, N, fs=128, freq_range=(0.5, 5.0), dropout=0.3):
        super().__init__()
        self.chancomb = ChannelCombination(C, T, dropout)
        self.fftfeat  = FFTFeatureExtractor(fs, T, freq_range)
        self.mlphead = MLPHead(C * self.fftfeat.output_dim, N, dropout)
        self.apply(_init_weights)

    def forward(self, x):  # x: (B, C, T)
        x = self.chancomb(x)                # (B, C, T)
        x = self.fftfeat(x)                 # (B, C, F_selected)
        x = x.flatten(start_dim=1)          # (B, C×F_selected)
        out = self.mlphead(x)               # (B, N)
        return out
