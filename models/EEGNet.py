# models/EEGNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module.util import Conv2dWithConstraint, LinearWithConstraint

class EEGNet(nn.Module):
    def __init__(self,
                 F1: int, D: int, F2: int,
                 kernel_size_1: tuple, kernel_size_2: tuple, kernel_size_3: tuple,
                 pooling_size_1: tuple, pooling_size_2: tuple,
                 dropout_rate: float,
                 signal_length: int, num_class: int):
        super().__init__()

        # —— Layer 1: temporal conv —— #
        k_h, k_w = kernel_size_1
        # compute manual padding: (left, right, top, bottom)
        pad_w = (k_w//2, k_w//2 - (1 if k_w%2==0 else 0))
        pad_h = (k_h//2, k_h//2 - (1 if k_h%2==0 else 0))
        self.pad1 = lambda x: F.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]))
        self.conv2d   = nn.Conv2d(1, F1, kernel_size=(k_h, k_w), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(F1)

        # —— Layer 2: depthwise conv over time —— #
        self.depthwise_conv2d = Conv2dWithConstraint(
            in_channels=F1,
            out_channels=D*F1,
            kernel_size=kernel_size_2,
            groups=F1, bias=False, max_norm=1.0
        )
        self.batch_norm2 = nn.BatchNorm2d(D*F1)
        self.elu        = nn.ELU()
        self.avg_pool1  = nn.AvgPool2d(kernel_size=pooling_size_1)
        self.dropout1   = nn.Dropout2d(p=dropout_rate)

        # —— Layer 3: separable conv —— #
        self.sep_conv_depth = nn.Conv2d(
            in_channels=D*F1,
            out_channels=D*F1,
            kernel_size=kernel_size_3,
            padding='same',
            groups=D*F1,
            bias=False
        )
        self.sep_conv_point = nn.Conv2d(D*F1, F2, kernel_size=(1,1), bias=False)
        self.batch_norm3 = nn.BatchNorm2d(F2)
        self.avg_pool2  = nn.AvgPool2d(kernel_size=pooling_size_2)
        self.dropout2   = nn.Dropout2d(p=dropout_rate)

        # —— Classifier —— #
        reduced_time = signal_length \
                       // pooling_size_1[1] \
                       // pooling_size_2[1]
        flat_dim = F2 * 1 * reduced_time
        self.flatten = nn.Flatten(start_dim=1)
        self.dense   = LinearWithConstraint(
            in_features=flat_dim, out_features=num_class, max_norm=0.25
        )

        # —— weight initialization —— #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        # allow x: (B,C,T)  or  (B,1,C,T)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # -> (B,1,C,T)

        # Layer1
        x = self.pad1(x)
        x = self.conv2d(x)
        x = self.batch_norm1(x)

        # Layer2
        x = self.depthwise_conv2d(x)
        x = self.batch_norm2(x)
        x = self.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        # Layer3
        x = self.sep_conv_depth(x)
        x = self.batch_norm3(self.sep_conv_point(x))
        x = self.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        # Classifier
        x = self.flatten(x)
        return self.dense(x)
