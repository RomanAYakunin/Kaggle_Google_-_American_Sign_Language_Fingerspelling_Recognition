import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import FeatureGenerator


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_lin = nn.Linear(dim, dim * 3)  # TODO combine into one linear layer
        self.scaling = math.sqrt(dim // num_heads)
        self.out_lin = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ELU()
        )

    def forward(self, x, mask):  # x.shape = [N, L, model_dim], mask.shape = [N, L]
        qkv = self.qkv_lin(x).reshape(x.shape[0], x.shape[1], self.num_heads, -1, 3)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]  # [N, L, num_heads, head_dim]
        q = q.permute(0, 2, 1, 3)  # [N, num_heads, L, head_dim]
        k = k.permute(0, 2, 3, 1)  # [N, num_heads, head_dim, L]
        v = F.elu(v).permute(0, 2, 1, 3)  # [N, num_heads, L, head_dim]
        attn = torch.matmul(q, k)  # [N, num_heads, L, L]
        mask = mask.unsqueeze(1).unsqueeze(2)  # [N, 1, 1, L]
        mask = torch.where(mask,
                           torch.full_like(mask, fill_value=-torch.inf, dtype=torch.float32),
                           torch.zeros_like(mask, dtype=torch.float32))  # [N, 1, 1, L]
        attn = F.softmax(attn / self.scaling + mask, dim=3)  # [N, num_heads, L, L]
        out = torch.matmul(attn, v)  # [N, num_heads, L, head_dim]
        out = out.permute(0, 2, 1, 3)  # [N, L, num_heads, head_dim]
        out = out.reshape(out.shape[0], out.shape[1], -1)  # [N, L, model_dim]
        out = self.out_lin(out)
        return out  # TODO try adding dropout somewhere here too


class EncoderLayer(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.sa_dropout = nn.Dropout(dropout)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self._sa_block(x, mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, mask):
        x = self.self_attn(x, mask)
        return self.sa_dropout(x)

    def _ff_block(self, x):  # TODO experiment with activation placement
        x = self.linear2(self.ff_dropout1(nn.functional.elu(self.linear1(x))))
        return self.ff_dropout2(x)


class Encoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers, dropout):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList([
            EncoderLayer(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        for layer in self.encoder:
            x = layer(x, mask)
        return x