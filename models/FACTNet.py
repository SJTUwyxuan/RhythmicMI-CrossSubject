# models/FACTNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.module.util import Conv2dWithConstraint, LinearWithConstraint


def get_frequency_modes(seq_len, modes=16, mode_select_method='segmented_random'):
    """
    Select frequency indices for frequency-domain modulation.
    """
    modes = min(modes, seq_len // 2)

    if mode_select_method == 'random':
        index = np.random.choice(seq_len // 2, modes, replace=False)

    elif mode_select_method == 'segmented_random':
        segments = np.array_split(np.arange(seq_len // 2), modes)
        index = [np.random.choice(segment) for segment in segments]

    else:
        index = np.arange(modes)

    index.sort()
    return index


class FA(nn.Module):
    """
    Frequency Attention (FA) module operating in Fourier domain.
    """

    def __init__(
        self,
        in_depth=1,
        in_channel=22,
        seq_len=640,
        modes=16,
        mode_select_method='segmented_random',
        dropout=0.3
    ):
        super().__init__()

        self.seq_len = seq_len
        self.radio = 1

        # Selected frequency modes
        self.index = get_frequency_modes(
            seq_len,
            modes=modes,
            mode_select_method=mode_select_method
        )

        # Learnable real and imaginary weights
        self.fweights_size = [min(modes, math.ceil(len(self.index) / self.radio)) + 1, 1]
        self.fweights = nn.Parameter(torch.zeros(self.fweights_size), requires_grad=True)
        self.fweights_im = nn.Parameter(torch.zeros(self.fweights_size), requires_grad=True)

        self.dropout = nn.Dropout(p=dropout)

    def compl_mul1d(self, input, weights, i):
        weight = weights[i].unsqueeze(-1).unsqueeze(-1)
        return input * weight

    def forward(self, x):
        x = x.squeeze(1)
        B, E, L = x.shape

        # FFT
        ffted = torch.fft.rfftn(x, dim=-1, norm='ortho')
        re, im = ffted.real, ffted.imag
        M = L // 2 + 1

        out_re = torch.zeros(B, E, M, device=x.device)
        out_im = torch.zeros(B, E, M, device=x.device)

        # Frequency-wise modulation
        for wi, idx in enumerate(self.index):
            widx = wi if wi == 0 else int(wi / self.radio) + 1
            out_re[:, :, wi] = self.compl_mul1d(re[:, :, idx], self.fweights, widx)
            out_im[:, :, wi] = self.compl_mul1d(im[:, :, idx], self.fweights_im, widx)

        # Weight normalization
        self.fweights.data = torch.renorm(self.fweights.data, p=2, dim=0, maxnorm=1)
        self.fweights_im.data = torch.renorm(self.fweights_im.data, p=2, dim=0, maxnorm=1)

        # Inverse FFT
        x_ft = torch.complex(re + out_re, im + out_im)
        x = torch.fft.irfftn(x_ft, s=L, dim=-1, norm='ortho')

        if x.dim() != 4:
            x = x.unsqueeze(1)

        return x


class EEGDepthAware(nn.Module):
    """
    Depth-aware spatial weighting across EEG channels.
    """

    def __init__(self, W, C, k=7):
        super().__init__()

        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(
            1, 1,
            kernel_size=(k, 1),
            padding=(k // 2, 0),
            bias=True
        )
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        x_pool = self.adaptive_pool(x)
        x_t = x_pool.transpose(-2, -3)
        y = self.conv(x_t)
        y = self.softmax(y)
        y = y.transpose(-2, -3)

        return y * self.C * x


class FFT_Based_Refactor(nn.Module):
    """
    Extract dominant periodic components via FFT magnitude ranking.
    """

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.top_list = None

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)

        # Average amplitude across batch and channels
        freq_amp = xf.abs().mean(0).mean(-1)
        freq_amp[0] = 0

        _, topk = torch.topk(freq_amp, self.k)
        topk = topk.detach().cpu().numpy()

        # Ensure base frequency is included
        for i in range(self.k - 1, 0, -1):
            topk[i] = topk[i - 1]
        topk[0] = 1

        self.top_list = topk

        xf = torch.fft.rfft(x, dim=1)
        period = x.shape[1] // self.top_list

        return period, xf.abs().mean(-1)[:, self.top_list]


class Multi_periodicity_Inception(nn.Module):
    """
    Inception-style convolution blocks for multiple periodic scales.
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=None):
        super().__init__()

        # Default kernel sizes covering different rhythmic periods
        if kernel_sizes is None:
            kernel_sizes = [15, 31, 63]

        layers = []
        for k in kernel_sizes:
            layers.append(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, k),
                padding=(0, k // 2),
                groups=in_channels,
                bias=False
            ))

        # Pooling branch
        layers.append(nn.AvgPool2d(kernel_size=(1, 3), padding=(0, 1)))

        self.branches = nn.ModuleList(layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        outs = [branch(x) for branch in self.branches[:-1]]

        # Average across branches
        return torch.stack(outs, dim=-1).mean(-1)


class TPI(nn.Module):
    """
    Temporal Periodicity Interaction module.
    """

    def __init__(self, F2, kernel_sizes=None, seq_len=40):
        super().__init__()

        self.seq_len = seq_len
        self.fft_ref = FFT_Based_Refactor(
            k=len(kernel_sizes) if kernel_sizes else 3
        )

        self.conv = nn.Sequential(
            Multi_periodicity_Inception(F2, F2, kernel_sizes=kernel_sizes),
            nn.GELU(),
            Multi_periodicity_Inception(F2, F2, kernel_sizes=kernel_sizes)
        )

    def forward(self, x):
        B, T, N = x.size()

        periods, weights = self.fft_ref(x)
        res = []

        for i, p in enumerate(periods):

            if self.seq_len % p != 0:
                length = (self.seq_len // p + 1) * p
                pad = torch.zeros(B, length - self.seq_len, N, device=x.device)
                out = torch.cat([x, pad], dim=1)
            else:
                length = self.seq_len
                out = x

            out = out.reshape(B, length // p, p, N)\
                     .permute(0, 3, 1, 2)\
                     .contiguous()

            out = self.conv(out)

            out = out.permute(0, 2, 3, 1)\
                     .reshape(B, -1, N)[:, :self.seq_len, :]

            res.append(out)

        w = F.softmax(weights, dim=1).unsqueeze(1).unsqueeze(1)
        res = torch.stack(res, dim=-1) * w

        return res.sum(-1) + x


class FACTNet(nn.Module):
    """
    FACTNet architecture for rhythmic EEG decoding.
    """

    def __init__(self, nChan=22, nTime=640, nClass=4):
        super().__init__()

        F0, F1, D, F2 = 1, 8, 2, 16
        seq_len = nTime // 16

        # Optional frequency attention
        self.use_fa = True # for SSMRR decoding, FA is not that neccessary
        if self.use_fa:
            self.fa = FA(
                in_depth=1,
                in_channel=nChan,
                seq_len=nTime,
                modes=16
            )

        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(F0, F1, (1, 128), padding=(0, 64), bias=False),
            nn.BatchNorm2d(F1)
        )

        # Channel-wise convolution
        self.channel_conv = nn.Sequential(
            Conv2dWithConstraint(
                F1, F1 * D, (nChan, 1),
                groups=F1,
                bias=False,
                max_norm=1.
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.3)
        )

        # Depthwise separable convolution
        self.depth_separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16),
                      padding=(0, 8),
                      groups=F1 * D,
                      bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.3)
        )

        # Depth-aware weighting
        self.depth_aware = EEGDepthAware(W=seq_len, C=F2, k=7)

        # Temporal periodic modeling
        kernel_sizes = [15, 31, 63]
        self.model = nn.ModuleList([
            TPI(F2, kernel_sizes=kernel_sizes, seq_len=seq_len)
        ])

        self.layer_norm = nn.LayerNorm(F2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(F2 * seq_len, nClass, max_norm=0.25),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):

        if x.dim() != 4:
            x = x.unsqueeze(1)

        if self.use_fa:
            x = self.fa(x)

        x = self.temporal_conv(x)
        x = self.channel_conv(x)
        x = self.depth_separable(x)
        x = self.depth_aware(x)

        x = x.squeeze(2).permute(0, 2, 1)

        for layer in self.model:
            x = self.layer_norm(layer(x))

        return self.classifier(x)
