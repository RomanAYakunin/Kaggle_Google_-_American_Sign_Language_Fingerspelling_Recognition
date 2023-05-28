import math
import torch
import torch.nn as nn
from dataset import FeatureGenerator
import torch.nn.functional as F
from copy import deepcopy


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
        FG = FeatureGenerator()
        self.num_heads = num_heads
        self.window_size = window_size
        self.dilation = dilation

        self.attn_lin = nn.Linear(dim, num_heads)
        self.pos_net = nn.Sequential(
            nn.Linear(dim, num_heads),
            nn.LayerNorm(num_heads),
            nn.ELU(),
            nn.Linear(num_heads, (window_size + 1) * num_heads)
        )
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
                         torch.arange(FG.max_len).unsqueeze(1)  # [max_len, window_size]
        indices_buffer -= dilation * (window_size // 2)
        indices_buffer = torch.where(indices_buffer < 0,
                                     torch.full_like(indices_buffer, fill_value=FG.max_len),
                                     indices_buffer)
        self.register_buffer('indices_buffer', indices_buffer)  # for extracting sliding windows

    def forward(self, x, mask):  # x: [N, L, in_dim], mask: [N, L]  # TODO remove t
        attn_exp = torch.exp(self.attn_lin(x)) * (~mask).to(torch.float32).unsqueeze(2)  # [N, L, num_heads]
        attn = attn_exp / (torch.sum(attn_exp, dim=1, keepdim=True) + 1e-5)
        v = self.v_lin(x)
        g_pool = torch.sum(attn.unsqueeze(3) * v.reshape(x.shape[0], x.shape[1], self.num_heads, -1), dim=1)
        g_pool = g_pool.reshape(x.shape[0], -1)  # [N, dim]
        pos_component = torch.exp(self.pos_net(g_pool).reshape(-1, 1, self.window_size + 1, self.num_heads))

        attn_win = self._extract_sliding_windows(attn_exp)  # [N, L, window_size, num_heads]
        attn_win = torch.cat([torch.ones(x.shape[0], x.shape[1], 1, self.num_heads, device=x.device), attn_win], dim=2)
        attn_win = attn_win * pos_component  # [N, L, window_size + 1, num_heads]
        attn_win = attn_win / (torch.sum(attn_win, dim=2, keepdim=True) + 1e-5)
        v = self._extract_sliding_windows(v)  # [N, L, window_size, out_dim]
        v = v.reshape(x.shape[0], x.shape[1], self.window_size, self.num_heads, -1)  # [N, L, window_size, num_heads, head_dim]

        out = torch.sum(attn_win[:, :, :self.window_size].unsqueeze(4) * v, dim=2)  # [N, L, num_heads, head_dim]
        out = out + \
              (attn_win[:, :, 0].unsqueeze(3) * g_pool.reshape(x.shape[0], 1, self.num_heads, -1))
        out = out.reshape(x.shape[0], x.shape[1], -1)  # [N, L, out_dim]
        out = self.out_lin(out)
        return out + x

    def _extract_sliding_windows(self, x):
        indices = self.indices_buffer[:x.shape[1]]
        indices = torch.minimum(indices, torch.full_like(indices, fill_value=x.shape[1]))
        x = torch.cat([x,
                       torch.zeros(x.shape[0], 1, x.shape[2], dtype=torch.float32, device=x.device)], dim=1)
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
        self.num_heads = 32

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
        self.sliding_attn1 = SlidingATTN(self.dim, num_heads=self.num_heads, window_size=9, dilation=1)
        self.sliding_attn2 = SlidingATTN(self.dim, num_heads=self.num_heads, window_size=9, dilation=3)
        self.sliding_attn3 = SlidingATTN(self.dim, num_heads=self.num_heads, window_size=9, dilation=3)
        self.sliding_attn4 = SlidingATTN(self.dim, num_heads=self.num_heads, window_size=9, dilation=3)
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
        sliding_attn4_out = self.sliding_attn4(sliding_attn3_out, mask)
        out = self.output_lin(sliding_attn4_out)
        return out