import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F


def _init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ChannelCombination(nn.Module):
    def __init__(self, in_channels, feature_len, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.ln = nn.LayerNorm([in_channels, feature_len])
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln(x)
        x = self.act(x)
        x = self.drop(x)
        return x.flatten(1)


class FreqExtracter(nn.Module):
    def __init__(self, T, fs, freq_range=(0.5, 5.0)):
        super().__init__()
        freqs = torch.fft.rfftfreq(T, d=1 / fs)
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        self.register_buffer("mask", mask)
        self.num_bins = int(mask.sum().item())

    def forward(self, x):
        X = torch.fft.rfft(x, dim=-1)
        return X[..., self.mask]
    
    
class FeatureExtracter(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super().__init__()
        hidden = 6 * num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            # nn.LayerNorm(hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class MLPHead(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super().__init__()
        fusionhidden = 6 * num_classes
        self.head = nn.Sequential(
            nn.Linear(input_dim, fusionhidden),
            nn.BatchNorm1d(fusionhidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusionhidden, num_classes)
        )

    def forward(self, x):
        return self.head(x)


class TimeFreqMLP(nn.Module):
    def __init__(self, C, T, N, fs=128, freq_range=(0.5, 5.0), dropout_freq=0.5, dropout_time=0.75):
        super().__init__()
        self.fft_extractor = FreqExtracter(T, fs, freq_range)

        F = self.fft_extractor.num_bins
        self.freq_comb = ChannelCombination(2 * C, F, dropout_freq)
        self.time_comb = ChannelCombination(C, T, dropout_time)

        self.dim_freq = 2 * C * F
        self.dim_time = C * T

        self.freqExtractor = FeatureExtracter(self.dim_freq, N, dropout_freq)
        self.timeExtractor = FeatureExtracter(self.dim_time, N, dropout_time)

        self.classifier = MLPHead(12 * N, N, dropout_freq)
        self.apply(_init_weights)

    def forward(self, x):
        # Freq branch
        fft = self.fft_extractor(x)
        real, imag = fft.real, fft.imag
        fft_feat = torch.cat([real, imag], dim=1)
        freq_out = self.freq_comb(fft_feat)
        freqfeature = self.freqExtractor(freq_out)

        # Time branch
        time_out = self.time_comb(x)
        timefeature = self.timeExtractor(time_out)

        # Fusion
        fusionfeature = torch.cat([freqfeature, timefeature], dim=1)

        return self.classifier(fusionfeature)
