import numpy as np
import math

import torch
from torch import nn
from torch.nn import functional as F



class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos


class self_attention(nn.Module):
    def __init__(self, dim, num_head=8, qkv_bias=False, attn_drop=0., proj_drop=0.) -> None:
        super().__init__()
        self.num_head = num_head

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.projection = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, C // self.num_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1)#TODO
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.projection(x)
        x = self.proj_drop(x)
        return x


class LGFI_Block(nn.Module):
    def __init__(self, dim, drop_path=0, layer_scale_init_value=1e-6, expan_rate=6,
                use_pos_embed=True, num_head=6, qkv_bias=True, atten_drop=0., drop=0.) -> None:
        super().__init__()

        self.dim = dim
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = PositionalEncodingFourier(dim = dim)
        
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = self_attention(dim, num_head=num_head, qkv_bias=qkv_bias, attn_drop=atten_drop, proj_drop=drop)

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_rate*dim)
        self.gelu = nn.GELU()
        self.pwconv2 = nn.Linear(expan_rate*dim, dim)
    
    def forward(self, x):
        input = x

        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W).permute(0, 2, 1)

        if self.pos_embed:
            pos_embed = self.pos_embed(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_embed
        

        x = x + self.self_attn(self.attn_norm(x))
        x = x.reshape(B, H, W, C)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.gelu(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        x = input + x
        return x





