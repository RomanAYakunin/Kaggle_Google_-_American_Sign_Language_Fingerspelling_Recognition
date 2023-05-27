import math
import torch
import torch.nn as nn
from dataset import FeatureGenerator
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()
        self.d_model = dim
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000) / dim))  # TODO try changing 10000
        angle_arr = position * div_term  # [max_len, dim / 2]
        pe_sin = torch.sin(angle_arr)
        pe_cos = torch.cos(angle_arr)  # TODO try different constant than 10_000
        pe = torch.stack([pe_sin, pe_cos], dim=2).reshape(max_len, -1)
        self.register_buffer('pe', pe)  # [max_len, dim]

    def forward(self, x):
        return x + self.pe[:x.shape[1]].unsqueeze(0)


class SlidingATTN(nn.Module):
    def __init__(self, dim, num_heads, window_size, dilation):  # window_size must be odd
        super(SlidingATTN, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.dilation = dilation

        self.attn_lin = nn.Linear(dim, num_heads)
        self.pos_component = nn.Parameter(torch.Tensor(torch.zeros(1, 1, window_size, num_heads)))
        nn.init.xavier_uniform_(self.pos_component)
        self.v_lin = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ELU()
        )
        self.out_lin = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        indices_buffer = dilation * torch.arange(window_size).unsqueeze(0) + \
                         torch.arange(FeatureGenerator().max_len).unsqueeze(1)
        self.register_buffer('indices_buffer', indices_buffer)  # for extracting sliding windows

    def forward(self, x, mask):  # x: [N, L, in_dim], mask: [N, L]  # TODO remove t
        attn = torch.exp(self.attn_lin(x)) * (~mask).to(torch.float32).unsqueeze(2)  # [N, L, num_heads]
        attn = self._extract_sliding_windows(attn) * torch.exp(self.pos_component)
        attn = attn / (torch.sum(attn, dim=2, keepdim=True) + 1e-5)

        v = self._extract_sliding_windows(self.v_lin(x))  # [N, L, window_size, out_dim]
        v = v.reshape(x.shape[0], x.shape[1], self.window_size, self.num_heads, -1)

        out = torch.sum(attn.unsqueeze(4) * v, dim=2)  # [N, L, num_heads, head_dim]
        out = out.reshape(x.shape[0], x.shape[1], -1)  # [N, L, out_dim]
        out = self.out_lin(out)
        return out + x

    def _extract_sliding_windows(self, x):
        padding = self.dilation * self.window_size // 2
        indices = self.indices_buffer[:x.shape[1]]
        x = torch.cat([torch.zeros(x.shape[0], padding, x.shape[2], dtype=torch.float32, device=x.device),
                       x,
                       torch.zeros(x.shape[0], padding, x.shape[2], dtype=torch.float32, device=x.device)], dim=1)
        x = x[:, indices]
        return x


class AxisLayerNorm(nn.Module):
    def __init__(self, num_points, num_axes, dim):
        super(AxisLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.empty(1, 1, num_points, num_axes))
        self.beta = nn.Parameter(torch.empty(1, 1, num_points, num_axes))
        nn.init.xavier_uniform_(self.gamma)
        nn.init.xavier_uniform_(self.beta)
        self.dim = dim

    def forward(self, x):  # [N, L, num_points, num_axes]
        weight = (x != 0).to(torch.float32)
        x = x - (torch.sum(x * weight, dim=self.dim, keepdim=True) /
                 (torch.sum(weight, dim=self.dim, keepdim=True) + 1e-5))
        x_std = torch.sqrt(torch.sum(torch.square(x) * weight, dim=self.dim, keepdim=True) /
                           (torch.sum(weight, dim=self.dim, keepdim=True) + 1e-5))
        x = x / (x_std + 1e-5)
        x = self.gamma * x + self.beta
        x = x * weight
        return x


class Model(nn.Module):  # TODO try copying hyperparams from transformer_branch
    def __init__(self):
        super(Model, self).__init__()
        FG = FeatureGenerator()
        self.num_points = FG.num_points
        self.num_axes = FG.num_axes
        self.norm_ranges = FG.norm_ranges

        self.x_norm = AxisLayerNorm(self.num_points, self.num_axes, (1, 2))
        self.feature_norms = nn.ModuleList([AxisLayerNorm(end - start, self.num_axes, 2)
                                            for start, end in self.norm_ranges])

        self.dim = 768
        self.num_heads = 128

        self.input_net = nn.Sequential(
            nn.Linear(2 * self.num_points * self.num_axes, 2 * self.dim),  # TODO try turning off bias
            nn.LayerNorm(2 * self.dim),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(2 * self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.pos_enc = PositionalEncoding(dim=self.dim, max_len=FG.max_len)
        self.sliding_attn1 = SlidingATTN(self.dim, num_heads=self.num_heads, window_size=5, dilation=1)
        self.sliding_attn2 = SlidingATTN(self.dim, num_heads=self.num_heads, window_size=5, dilation=3)
        self.sliding_attn3 = SlidingATTN(self.dim, num_heads=self.num_heads, window_size=5, dilation=3)
        self.output_lin = nn.Linear(self.dim, 60)

    def forward(self, x):  # [N, L, num_points, num_axes]
        mask = torch.all(torch.all(x == 0, dim=2), dim=2)  # [N, L]
        normed_x = self.x_norm(x)
        normed_features = torch.cat([self.feature_norms[i](x[:, :, start: end])  # TODO make use of symmetry?
                                     for i, (start, end) in enumerate(self.norm_ranges)], dim=2)
        x = torch.cat([normed_x, normed_features], dim=2)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        input_net_out = self.pos_enc(self.input_net(x))
        sliding_attn1_out = self.sliding_attn1(input_net_out, mask)
        sliding_attn2_out = self.sliding_attn2(sliding_attn1_out, mask)
        sliding_attn3_out = self.sliding_attn3(sliding_attn2_out, mask)
        out = self.output_lin(sliding_attn3_out)
        return out