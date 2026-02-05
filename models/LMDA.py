# models/LMDA.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGDepthAttention(nn.Module):
    """
    Depth Attention module in the original LMDA-Net.

    Input:
        x: (B, d1, C, W)

    Steps:
        1) Adaptive pooling to (B, d1, 1, W)
        2) Transpose to (B, 1, d1, W)
        3) 2D convolution over d1 + softmax to obtain attention weights
        4) Restore shape and apply attention to x (with broadcasting)
    """

    def __init__(self, W: int, C: int, k: int = 7):
        super().__init__()
        # W: temporal width before attention (in samples)
        # C: number of feature channels after time_conv (i.e., d1)
        # k: kernel size along d1; should be odd and <= C
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(k, 1),
            padding=(k // 2, 0),
            bias=True
        )
        # Softmax along the d1 dimension
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x: (B, d1, C, W)
        x_pool = self.adaptive_pool(x)        # (B, d1, 1, W)
        x_t = x_pool.transpose(1, 2)          # (B, 1, d1, W)
        y = self.conv(x_t)                    # (B, 1, d1, W)
        y = self.softmax(y)                   # softmax over d1
        y = y.transpose(1, 2)                 # (B, d1, 1, W)

        # Broadcast and scale by C (as in the original implementation)
        return y * self.C * x


class LMDA(nn.Module):
    """
    Original LMDA-Net with five stages:
        1) Channel-wise weighting replication (depth branches)
        2) Temporal separable convolution
        3) Depth attention
        4) Spatial separable convolution + pooling
        5) Flatten + linear classification
    """

    def __init__(
        self,
        chans: int,
        samples: int,
        num_classes: int,
        depth: int = 9,
        kernel: int = 75,
        channel_depth1: int = 24,
        channel_depth2: int = 9,
        avepool: int = 5
    ):
        super().__init__()

        # (1) Learnable channel weights: (depth, 1, chans)
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans))
        nn.init.xavier_uniform_(self.channel_weight)

        # (2) Temporal separable convolution
        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), bias=False),  # pointwise
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(  # depthwise along time
                channel_depth1,
                channel_depth1,
                kernel_size=(1, kernel),
                padding=(0, kernel // 2),
                groups=channel_depth1,
                bias=False
            ),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )

        # (4) Spatial separable convolution (pointwise, then depthwise across channels)
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(
                channel_depth2,
                channel_depth2,
                kernel_size=(chans, 1),
                groups=channel_depth2,
                bias=False
            ),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        # Pooling + dropout
        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),  # temporal downsampling
            nn.Dropout(0.65),
        )

        # (3) Depth attention
        # Note: k should be an odd number <= channel_depth1 (default 7 in the original paper/code).
        self.depthAttention = EEGDepthAttention(
            W=samples,
            C=channel_depth1,
            k=7
        )

        # Infer flattened feature dimension for the classifier
        with torch.no_grad():
            dummy = torch.ones(1, 1, chans, samples)
            x = torch.einsum('bdcw,hdc->bhcw', dummy, self.channel_weight)
            x = self.time_conv(x)
            x = self.depthAttention(x)
            x = self.chanel_conv(x)
            x = self.norm(x)
            flat_dim = x.flatten(1).size(1)

        # Classification head
        self.classifier = nn.Linear(flat_dim, num_classes)

        # Parameter initialization (consistent with the original implementation)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, C, T) or (B, 1, C, T)
        """
        # Normalize input shape to (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # (1) Channel-weighted replication -> (B, depth, C, T)
        x = torch.einsum('bdcw,hdc->bhcw', x, self.channel_weight)

        # (2) Temporal separable conv -> (B, d1, C, T)
        x = self.time_conv(x)

        # (3) Depth attention -> (B, d1, C, T)
        x = self.depthAttention(x)

        # (4) Spatial separable conv + pooling -> (B, d2, 1, T/avepool)
        x = self.chanel_conv(x)
        x = self.norm(x)

        # (5) Flatten + classification
        feats = x.flatten(1)
        return self.classifier(feats)
