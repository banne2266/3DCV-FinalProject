import numpy as np

import torch
from torch import nn
from torch.nn import functional as F



class CDC_Block(nn.Module):
    def __init__(self, dim, k, dilation=1, stride=1, expand_rate = 6) -> None:
        super().__init__()
        padding = int((k - 1) / 2) * dilation
        self.D_conv = nn.Conv2d(dim, dim, k, stride=stride, padding=padding, dilation=dilation, groups=dim)
        self.bn = nn.BatchNorm2d(dim)

        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expand_rate*dim)
        self.gelu = nn.GELU()
        self.pwconv2 = nn.Linear(expand_rate*dim, dim)
    
    def forward(self, x):
        input = x

        x = self.D_conv(x)
        x = self.bn(x)

        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.gelu(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        x = x + input
        return x