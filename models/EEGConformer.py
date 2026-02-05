# models/EEGConformer.py

import sys
sys.path.insert(0, '/data2/wyxuan/Transfer-Learning/Transfe_Learning_2')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    """
    将 (B, C, T) 先 unsqueeze -> (B,1,C,T)，再做浅层卷积 + 池化，
    最后投影成序列 (B, L, E)。
    """
    def __init__(self, in_ch, emb_size, kernel_time, pool_time, pool_stride, dropout):
        super().__init__()
        # 浅层卷积：1×kernel_time
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, emb_size, kernel_size=(1, kernel_time), padding=(0, kernel_time//2)),
            nn.Conv2d(emb_size, emb_size, kernel_size=(in_ch, 1), padding=(0, 0)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d((1, pool_time), stride=(1, pool_stride)),
            # nn.AvgPool2d((1, pool_time)),
            nn.Dropout(dropout),
        )
        # (B, E, C, W) -> (B, W*C, E)
        self.projection = Rearrange('b e c w -> b (c w) e')

    def forward(self, x):
        # x: (B, C, T)
        x = x.unsqueeze(1)       # -> (B,1,C,T)
        x = self.shallownet(x)   # -> (B, E, C, W)
        x = self.projection(x)   # -> (B, L=W*C, E)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size, self.num_heads = emb_size, num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3, bias=False)
        self.att_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        # x: (B, L, E)
        B, L, E = x.shape
        qkv = self.qkv(x)                    # -> (B, L, 3E)
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, L, E) -> (B, h, L, d)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        # scaled dot-product attention
        scale = math.sqrt(E)
        att = torch.einsum('bhqd,bhkd->bhqk', q, k) / scale
        att = F.softmax(att, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhqk,bhkd->bhqd', att, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)

class FeedForward(nn.Sequential):
    def __init__(self, emb_size, expansion, dropout):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(dropout),
        )

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout, ff_expansion):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(emb_size),
            MultiHeadSelfAttention(emb_size, num_heads, dropout),
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(emb_size),
            FeedForward(emb_size, ff_expansion, dropout),
        )

    def forward(self, x):
        # self-attention + residual
        y = self.attn(x)
        x = x + y
        # feed-forward + residual
        y = self.ff(x)
        x = x + y
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size, num_heads, dropout, ff_expansion):
        super().__init__()
        self.layers = nn.Sequential(*[
            TransformerEncoderBlock(emb_size, num_heads, dropout, ff_expansion)
            for _ in range(depth)
        ])

    def forward(self, x):
        return self.layers(x)

class ClassificationHead(nn.Module):
    def __init__(self, emb_size, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            # 全局平均池化
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        # x: (B, L, E)
        return self.head(x)

class EEGConformer(nn.Module):
    """
    EEG Conformer: CNN + Transformer 用于 EEG 解码。
    输入 x: (B, C, T)  → 输出 logits: (B, N)
    """
    def __init__(self,
                 C, T, N,
                 emb_size=40,
                 depth=6,
                 num_heads=8,
                 ff_expansion=4,
                 dropout=0.5,
                 kernel_time=25,
                 pool_time=75,
                 pool_stride=15):
        super().__init__()
        # 1) patch embedding
        self.patch = PatchEmbedding(
            in_ch=C,
            emb_size=emb_size,
            kernel_time=kernel_time,
            pool_time=pool_time,
            pool_stride=pool_stride,
            dropout=dropout
        )
        # 2) transformer stack
        self.encoder = TransformerEncoder(
            depth=depth,
            emb_size=emb_size,
            num_heads=num_heads,
            dropout=dropout,
            ff_expansion=ff_expansion
        )
        # 3) classification head
        self.head = ClassificationHead(
            emb_size=emb_size,
            num_classes=N
        )
        # 权重初始化
        self.apply(self._init_weights)

    def forward(self, x):
        # x: (B, C, T)
        x = self.patch(x)      # -> (B, L, E)
        x = self.encoder(x)    # -> (B, L, E)
        return self.head(x)    # -> (B, N)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
