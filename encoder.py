import math
import torch
import torch.nn as nn
from dataset import FeatureGenerator
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.checkpoint import checkpoint


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()
        self.d_model = dim
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000) / dim))  # TODO try changing 10000
        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)

    def forward(self, x_len):
        angle_arr = self.position[:x_len] * self.div_term  # [max_len, dim / 2]
        pe_sin = torch.sin(angle_arr)
        pe_cos = torch.cos(angle_arr)
        pe = torch.stack([pe_sin, pe_cos], dim=2).reshape(1, x_len, -1)
        return pe


class SlidingATTN(nn.Module):
    def __init__(self, dim, num_heads, window_size, dilation, use_checkpoints=True):  # window_size must be odd
        super(SlidingATTN, self).__init__()
        self.use_checkpoints = use_checkpoints
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

    def forward(self, x, mask):  # x: [N, L, in_dim], mask: [N, L]
        attn_exp = torch.exp(self.attn_lin(x)) * (~mask).to(x.dtype).unsqueeze(2)  # [N, L, num_heads]
        attn = attn_exp / (torch.sum(attn_exp, dim=1, keepdim=True) + 1e-5)
        v = self.v_lin(x)
        g_pool = torch.sum(attn.unsqueeze(3) * v.reshape(x.shape[0], x.shape[1], self.num_heads, -1), dim=1)
        g_pool = g_pool.reshape(x.shape[0], -1)  # [N, dim]
        pos_component = torch.exp(self.pos_net(g_pool).reshape(-1, 1, self.window_size + 1, self.num_heads))

        if self.use_checkpoints:
            out = checkpoint(self.checkpoint_fn, attn_exp, v, g_pool, pos_component)
        else:
            out = self.checkpoint_fn(attn_exp, v, g_pool, pos_component)
        out = out.reshape(x.shape[0], x.shape[1], -1)  # [N, L, out_dim]
        out = self.out_lin(out)
        return out + x

    def checkpoint_fn(self, attn_exp, v, g_pool, pos_component):
        attn_win = self.extract_sliding_windows(attn_exp)  # [N, L, window_size, num_heads]
        attn_win = torch.cat([torch.ones(v.shape[0], v.shape[1], 1, self.num_heads, device=v.device), attn_win], dim=2)
        attn_win = attn_win * pos_component  # [N, L, window_size + 1, num_heads]
        attn_win = attn_win / (torch.sum(attn_win, dim=2, keepdim=True) + 1e-5)
        v = self.extract_sliding_windows(v)  # [N, L, window_size, out_dim]
        v = v.reshape(v.shape[0], v.shape[1], self.window_size, self.num_heads,
                      -1)  # [N, L, window_size, num_heads, head_dim]

        out = torch.sum(attn_win[:, :, 1:].unsqueeze(4) * v, dim=2)  # [N, L, num_heads, head_dim]
        out = out + \
              (attn_win[:, :, 0].unsqueeze(3) * g_pool.reshape(v.shape[0], 1, self.num_heads, -1))
        return out

    def extract_sliding_windows(self, x):
        indices = self.indices_buffer[:x.shape[1]]
        indices = torch.minimum(indices, torch.full_like(indices, fill_value=x.shape[1]))
        x = torch.cat([x,
                       torch.zeros(x.shape[0], 1, x.shape[2], dtype=x.dtype, device=x.device)], dim=1)
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
        weight = (x != 0).to(x.dtype)
        x = x - (torch.sum(x * weight, dim=self.dim, keepdim=True) /
                 (torch.sum(weight, dim=self.dim, keepdim=True) + 1e-5))
        x_std = torch.sqrt(torch.sum(torch.square(x) * weight, dim=self.dim, keepdim=True) /
                           (torch.sum(weight, dim=self.dim, keepdim=True) + 1e-5))
        x = x / (x_std + 1e-5)
        x = self.gamma * x + self.beta
        x = x * weight
        return x


class Encoder(nn.Module):
    def __init__(self, num_dec_layers, dec_dim, use_checkpoints=True):
        super(Encoder, self).__init__()
        FG = FeatureGenerator()
        self.num_points = FG.num_points
        self.num_axes = FG.num_axes
        self.norm_ranges = FG.norm_ranges
        self.lh_range, self.rh_range = FG.norm_ranges[-2], FG.norm_ranges[-1]

        self.x_norm = AxisLayerNorm(self.num_points, self.num_axes, (1, 2))
        self.feature_norms = nn.ModuleList([AxisLayerNorm(end - start, self.num_axes, 2)
                                            for start, end in self.norm_ranges])

        self.input_dim = 2 * self.num_points * self.num_axes
        self.dim = 896
        self.num_heads = 128

        self.input_net = nn.Sequential(
            nn.Linear(self.input_dim, 2 * self.dim),  # TODO try turning off bias
            nn.LayerNorm(2 * self.dim),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(2 * self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ELU(),
            nn.Dropout(0.5)
        )
        self.pos_enc = PositionalEncoding(dim=self.dim, max_len=FG.max_len)
        self.sliding_attn1 = SlidingATTN(self.dim, num_heads=self.num_heads, window_size=5, dilation=1,
                                         use_checkpoints=use_checkpoints)
        self.sliding_attn_stack = nn.ModuleList([
            SlidingATTN(self.dim, num_heads=self.num_heads, window_size=5, dilation=3,
                        use_checkpoints=use_checkpoints) for _ in range(5)
        ])

        self.num_dec_layers = num_dec_layers
        self.decoder_dim = dec_dim
        self.out_dim = self.num_dec_layers * 2 * self.decoder_dim

        self.out_lin = nn.Sequential(
            nn.Linear(self.dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ELU()
        )

    def forward(self, x, mask):  # x: [N, L, num_points, num_axes], mask: [N, L]
        normed_x = self.x_norm(x)
        normed_features = torch.cat([self.feature_norms[i](x[:, :, start: end])  # TODO make use of symmetry?
                                     for i, (start, end) in enumerate(self.norm_ranges)], dim=2)
        x = torch.cat([normed_x, normed_features], dim=2).flatten(start_dim=2)

        input_net_out = self.input_net(x) + self.pos_enc(x.shape[1])
        sliding_attn_out = self.sliding_attn1(input_net_out, mask)
        for sliding_attn in self.sliding_attn_stack:
            sliding_attn_out = sliding_attn(sliding_attn_out, mask)
        out = self.out_lin(sliding_attn_out).reshape(x.shape[0], -1, self.decoder_dim, self.num_dec_layers, 2)
        return out


# class ConvBlock(nn.Module):
#     def __init__(self, dim, window_size, dilation):  # window_size must be odd
#         super(ConvBlock, self).__init__()
#         FG = FeatureGenerator()
#         self.window_size = window_size
#         self.dilation = dilation
#
#         self.lin1 = nn.Sequential(
#             nn.Linear(window_size * dim, dim),
#             nn.LayerNorm(dim),
#             nn.ELU()
#         )
#         self.lin2 = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.LayerNorm(dim),
#             nn.ELU()
#         )
#
#         indices_buffer = dilation * torch.arange(window_size).unsqueeze(0) + \
#                          torch.arange(FG.max_len).unsqueeze(1)  # [max_len, window_size]
#         indices_buffer -= dilation * (window_size // 2)
#         indices_buffer = torch.where(indices_buffer < 0,
#                                      torch.full_like(indices_buffer, fill_value=FG.max_len),
#                                      indices_buffer)
#         self.register_buffer('indices_buffer', indices_buffer)  # for extracting sliding windows
#
#     def forward(self, x, mask):  # x: [N, L, dim], mask: [N, L]  # TODO remove t
#         inv_mask = (~mask).to(x.dtype).unsqueeze(2)
#         x = x * inv_mask
#         global_pool = torch.sum(x, dim=1, keepdim=True) / torch.sum(inv_mask, dim=1, keepdim=True)
#         win_x = self._extract_sliding_windows(x).reshape(x.shape[0], x.shape[1], -1)
#         out = self.lin2(self.lin1(win_x) + global_pool) + x
#         return out
#
#     def _extract_sliding_windows(self, x):
#         indices = self.indices_buffer[:x.shape[1]]
#         indices = torch.minimum(indices, torch.full_like(indices, fill_value=x.shape[1]))
#         x = torch.cat([x,
#                        torch.zeros(x.shape[0], 1, x.shape[2], dtype=x.dtype, device=x.device)], dim=1)
#         x = x[:, indices]
#         return x