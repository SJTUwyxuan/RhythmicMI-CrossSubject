import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

# === Multi-scale temporal-spectral convolutional transformer blocks ===
class Multi_Scale_Convformer(nn.Module):
    def __init__(self, token_num, token_length, kernal_length, dropout):
        super().__init__()
        self.att2conv = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=kernal_length, padding=kernal_length // 2),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.att2conv2 = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=int(kernal_length * 0.5),
                      padding=int(kernal_length * 0.5) // 2),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.att2conv3 = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=int(kernal_length * 0.25),
                      padding=int(kernal_length * 0.25) // 2),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.att2conv(x) + self.att2conv2(x) + self.att2conv3(x)

class Convformer(nn.Module):
    def __init__(self, token_num, token_length, kernal_length, dropout):
        super().__init__()
        self.att2conv = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=kernal_length, padding=kernal_length // 2),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.att2conv(x)

class Multi_Level_Convformer(nn.Module):
    def __init__(self, token_num, token_length, kernal_length, dropout):
        super().__init__()
        self.att2conv1 = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=int(kernal_length * 0.5), padding=int(kernal_length * 0.5) // 2),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.att2conv2 = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=kernal_length, padding=kernal_length // 2),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x1 = self.att2conv1(x)
        x2 = self.att2conv2(x1)
        return x1 + x2

# === Residual blocks with convolutional transformer and MLP fusion ===
class Block_fusion(nn.Module):
    def __init__(self, dim, token_num, kernal_length=31, dropout=0.5,
                 mlp_ratio=2., drop_path=0., norm_layer=nn.LayerNorm,
                 init_values=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Multi_Scale_Convformer(token_num, dim, kernal_length, dropout)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )
        self.gamma = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) * self.gamma
        x = x + self.drop_path(self.mlp(self.norm2(x))) * self.gamma
        return x

class Block_fre(nn.Module):
    def __init__(self, dim, token_num, kernal_length=31, dropout=0.5,
                 mlp_ratio=2., drop_path=0., norm_layer=nn.LayerNorm,
                 init_values=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Convformer(token_num, dim, kernal_length, dropout)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )
        self.gamma = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) * self.gamma
        x = x + self.drop_path(self.mlp(self.norm2(x))) * self.gamma
        return x

class Block_time(nn.Module):
    def __init__(self, dim, token_num, kernal_length=31, dropout=0.5,
                 mlp_ratio=2., drop_path=0., norm_layer=nn.LayerNorm,
                 init_values=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Multi_Level_Convformer(token_num, dim, kernal_length, dropout)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )
        self.gamma = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) * self.gamma
        x = x + self.drop_path(self.mlp(self.norm2(x))) * self.gamma
        return x

# === MTSNet wrapper for time+freq fusion ===
class MTSNet(nn.Module):
    def __init__(self, C, T, N, depth_L=2, depth_M=1,
                 kernal_length=128, dropout=0.3):
        super().__init__()
        # time-domain branch
        self.to_patch_time = nn.Sequential(
            nn.Conv1d(C, C*2, 1),
            nn.LayerNorm(T),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # freq-domain branch (magnitude)
        self.to_patch_fre = nn.Sequential(
            nn.Conv1d(C, C*2, 1),
            nn.LayerNorm(T//2+1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # build transformer stacks
        self.transformer_time = nn.Sequential(*[
            Block_time(dim=T, token_num=C*2, kernal_length=kernal_length, dropout=dropout)
            for _ in range(depth_L)
        ])
        self.transformer_fre = nn.Sequential(*[
            Block_fre(dim=T//2+1, token_num=C*2, kernal_length=kernal_length, dropout=dropout)
            for _ in range(depth_L)
        ])
        self.transformer_fusion = nn.Sequential(*[
            Block_fusion(dim=T + (T//2+1), token_num=C*2, kernal_length=kernal_length, dropout=dropout)
            for _ in range(depth_M)
        ])
        # classification head
        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear((T + (T//2+1)) * (C*2), N)
        )

    def forward(self, x):
        # x: (B, C, T)
        xt = self.to_patch_time(x)
        xt = self.transformer_time(xt)
        Xf = torch.fft.rfft(x, dim=-1)
        xf = torch.abs(Xf)
        xf = self.to_patch_fre(xf)
        xf = self.transformer_fre(xf)
        xcat = torch.cat([xt, xf], dim=-1)
        x = self.transformer_fusion(xcat)
        return self.mlp_head(x)
