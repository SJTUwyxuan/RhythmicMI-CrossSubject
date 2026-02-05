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
    def __init__(self, Cin, F, dropout=0.5):
        super().__init__()
        # 1x1 conv mixes channels
        self.conv1 = nn.Conv1d(Cin, Cin, kernel_size=1)
        self.ln1 = nn.LayerNorm([Cin, F])
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, Cin, F)
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, Cin, F, N, dropout=0.5, head_type="mlp"):
        super().__init__()
        in_dim = Cin * F

        head_type = head_type.lower()
        assert head_type in ["mlp", "linear"], "head_type must be 'mlp' or 'linear'"

        if head_type == "linear":
            self.net = nn.Sequential(
                nn.Linear(in_dim, N)
            )
        else:
            hidden = 6 * N
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, N),
            )

    def forward(self, x):
        b, c, f = x.shape
        x = x.reshape(b, c * f)
        return self.net(x)


class FreqMLP(nn.Module):
    """
    Ablations:
      - use_imag=False: remove imag part (real-only)
      - use_chancomb=False: remove ChannelCombination module (identity)
      - head_type="linear": remove MLP (use linear classifier)
    """
    def __init__(
        self,
        C,
        T,
        N,
        fs,
        freq_range=None,
        dropout=0.5,
        use_imag=True,
        use_chancomb=True,
        head_type="mlp",
    ):
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

        self.use_imag = bool(use_imag)
        self.use_chancomb = bool(use_chancomb)

        # Channel dimension after spectrum stacking
        Cin = (2 * C) if self.use_imag else C
        self.Cin = Cin
        self.F = F

        # Optional channel mixing block
        if self.use_chancomb:
            self.chancomb = ChannelCombination(Cin, F, dropout)
        else:
            self.chancomb = nn.Identity()

        # Head: mlp (default) or linear
        self.head = MLPHead(Cin, F, N, dropout, head_type=head_type)

        self.apply(_init_weights)

    def forward(self, x):
        # x: (B, C, T)
        Xf = torch.fft.rfft(x, dim=-1)  # (B, C, F_full)
        if self.freq_idx is not None:
            Xf = Xf[..., self.freq_idx]  # (B, C, F)

        real = Xf.real
        if self.use_imag:
            imag = Xf.imag
            z = torch.cat([real, imag], dim=1)  # (B, 2C, F)
        else:
            z = real  # (B, C, F)

        z = self.chancomb(z)  # (B, Cin, F) or identity
        out = self.head(z)     # (B, N)
        return out
