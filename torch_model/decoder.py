import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, use_checkpoints=True):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_checkpoints = use_checkpoints

        self.scaling = math.sqrt(dim // num_heads)
        self.out_lin = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ELU(),
            nn.Dropout(0.5)  # DROPOUT !!!
        )

    def forward(self, q, k, v, mask):  # q: [N, Lq, dim], k: [N, Lk, dim], v: [N, Lv, dim], mask: [..., ..., Lq, Lk]
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, self.dim // self.num_heads)  # [N, Lq, num_heads, head_dim]
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, self.dim // self.num_heads)  # [N, Lk, num_heads, head_dim]
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, self.dim // self.num_heads)  # [N, Lk, num_heads, head_dim]
        q = q.permute(0, 2, 1, 3)  # [N, num_heads, Lq, head_dim]
        k = k.permute(0, 2, 3, 1)  # [N, num_heads, head_dim, Lk]
        v = v.permute(0, 2, 1, 3)  # [N, num_heads, Lk, head_dim]
        if self.use_checkpoints:
            out = checkpoint(self.checkpoint_fn, q, k, v, mask)
        else:
            out = self.checkpoint_fn(q, k, v, mask)
        out = out.permute(0, 2, 1, 3)  # [N, Lq, num_heads, head_dim]
        out = out.reshape(out.shape[0], out.shape[1], out.shape[2] * out.shape[3])  # [N, Lq, model_dim]
        out = self.out_lin(out)
        return out

    def checkpoint_fn(self, q, k, v, mask):
        attn = torch.matmul(q, k)  # [N, num_heads, Lq, Lk]
        attn = F.softmax(attn / self.scaling + mask, dim=3)  # [N, num_heads, Lq, Lk]
        out = torch.matmul(attn, v)  # [N, num_heads, Lq, head_dim]
        return out


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, use_checkpoints=True):
        super(SelfAttention, self).__init__()
        self.qkv_lin = nn.Linear(dim, dim * 3)
        self.mha = MultiHeadAttention(dim, num_heads, use_checkpoints)

    def forward(self, x, mask):  # [N, L, dim]
        qkv = self.qkv_lin(x).reshape(x.shape[0], x.shape[1], x.shape[2], 3)
        q, k, v = qkv[..., 0], qkv[..., 1], F.elu(qkv[..., 2])
        return self.mha(q, k, v, mask) + x

    def infer_step(self, x, mask, kv_cache, idx):  # x: [N, dim], kv_cache: [N, L, dim, 2], idx: int
        qkv = self.qkv_lin(x).reshape(x.shape[0], x.shape[1], 3)
        q, k, v = qkv[..., 0], qkv[..., 1], F.elu(qkv[..., 2])
        kv_cache[:, idx, :, 0] = k
        kv_cache[:, idx, :, 1] = v
        x = self.mha(q.unsqueeze(1), kv_cache[:, :idx + 1, :, 0], kv_cache[:, :idx + 1, :, 1], mask).squeeze(1) + x
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, use_checkpoints=True):
        super(CrossAttention, self).__init__()
        self.q_lin = nn.Linear(dim, dim)
        self.mha = MultiHeadAttention(dim, num_heads, use_checkpoints)

    def forward(self, x, k, v, mask):  # [N, L, dim]
        q = self.q_lin(x)
        return self.mha(q, k, v, mask) + x


class FFNet(nn.Module):
    def __init__(self, dim):
        super(FFNet, self).__init__()
        self.net = nn.Sequential(  # TODO try shrinking
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ELU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.net(x) + x


class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, use_checkpoints=True):
        super(DecoderLayer, self).__init__()
        self.self_attn = SelfAttention(dim, num_heads, use_checkpoints)
        self.cross_attn = CrossAttention(dim, num_heads, use_checkpoints)
        self.ff_net = FFNet(dim)

    def forward(self, x, k, v, causal_mask, pad_mask):
        x = self.self_attn(x, causal_mask)
        x = self.cross_attn(x, k, v, pad_mask)
        x = self.ff_net(x)
        return x

    def infer_step(self, x, k, v, causal_mask, pad_mask, kv_cache, idx):
        # kv_cache: [N, Lp, dim, 2]
        x = self.self_attn.infer_step(x, causal_mask, kv_cache, idx)
        x = self.cross_attn(x.unsqueeze(1), k, v, pad_mask).squeeze(1)
        x = self.ff_net(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, dim, num_heads, use_checkpoints=True):
        super(Decoder, self).__init__()  # TODO remove padding token
        self.embedding = nn.Embedding(num_embeddings=63, embedding_dim=dim)
        # token <59 = phrase, 59 = stop, 60 = train SOT, 61 = supp SOT, 62 = pad, 63 = gislr SOT, >63 = gislr sign
        self.layers = nn.ModuleList([DecoderLayer(dim, num_heads, use_checkpoints) for _ in range(num_layers)])
        self.out_lin = nn.Linear(dim, 60)

    def forward(self, enc_out, tokens, token_pe, pad_mask):  # TODO consider using checkpoints
        # enc_out: [N, Lx, dim, num_layers, 2], tokens: [N, Lp], pad_mask: [N, Lx]
        x = self.embedding(tokens) + token_pe  # [N, Lp, dim]
        causal_mask = torch.triu(torch.full((tokens.shape[1], tokens.shape[1]),
                                            fill_value=-torch.inf, dtype=enc_out.dtype, device=enc_out.device),
                                 diagonal=1).unsqueeze(0).unsqueeze(1)  # [1, 1, Lp, Lp]
        pad_mask = torch.where(pad_mask,
                               torch.full_like(pad_mask, fill_value=-torch.inf,
                                               dtype=enc_out.dtype, device=enc_out.device),
                               torch.zeros_like(pad_mask, dtype=enc_out.dtype, device=enc_out.device))
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [N, 1, 1, Lx]
        for i, layer in enumerate(self.layers):
            k, v = enc_out[..., i, 0], enc_out[..., i, 1]
            x = layer(x, k, v, causal_mask, pad_mask)
        return self.out_lin(x)  # [N, Lp, 60]

    def infer_step(self, enc_out, tokens, token_pe, pad_mask, kv_cache, idx):
        # enc_out: [N, Lx, dim, num_layers, 2], tokens: [N, Lp], pad_mask: [N, Lx], kv_cache: [N, Lp, dim, num_layers, 2]
        x = self.embedding(tokens[:, idx]) + token_pe[:, idx]  # [N, dim]
        causal_mask = torch.zeros((1, 1, 1, 1), dtype=x.dtype, device=x.device)
        inf_pad_mask = torch.where(pad_mask,  # TODO move this to Model module
                                   torch.full_like(pad_mask, fill_value=-torch.inf,
                                                   dtype=enc_out.dtype, device=enc_out.device),
                                   torch.zeros_like(pad_mask, dtype=enc_out.dtype, device=enc_out.device))
        inf_pad_mask = inf_pad_mask.unsqueeze(1).unsqueeze(2)  # [N, 1, 1, Lx]
        for i, layer in enumerate(self.layers):
            k, v = enc_out[..., i, 0], enc_out[..., i, 1]
            x = layer.infer_step(x, k, v, causal_mask, inf_pad_mask, kv_cache[..., i, :], idx)
        x = self.out_lin(x)
        tokens[:, idx + 1] = torch.argmax(x, dim=-1)
