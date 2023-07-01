import math
import torch
import torch.nn as nn
from dataset import FeatureGenerator
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
from encoder import PositionalEncoding, Encoder
from decoder import Decoder


class Model(nn.Module):
    def __init__(self, use_checkpoints=True):
        super(Model, self).__init__()
        self.num_dec_layers = 3
        self.dec_dim = 384
        self.num_dec_heads = 32
        self.max_dec_len = 45
        self.enc = Encoder(self.num_dec_layers, self.dec_dim, use_checkpoints)
        self.token_pos_enc = PositionalEncoding(dim=self.dec_dim, max_len=self.max_dec_len)
        self.dec = Decoder(self.num_dec_layers, self.dec_dim, self.num_dec_heads)

    def forward(self, x, tokens):
        pad_mask = torch.all(torch.all(x == 0, dim=3), dim=2)  # [N, L]
        enc_out = self.enc(x, pad_mask)
        token_pe = self.token_pos_enc(tokens.shape[1])
        dec_out = self.dec(enc_out, tokens, token_pe, pad_mask)
        return dec_out

    def infer(self, x, sot):
        pad_mask = torch.all(torch.all(x == 0, dim=3), dim=2)  # [N, L]
        enc_out = self.enc(x, pad_mask)
        tokens = torch.full((x.shape[0], self.max_dec_len), fill_value=62, dtype=torch.long, device=x.device)
        if sot == 'train':
            sot = 60
        if sot == 'supp':
            sot = 61
        tokens[:, 0] = sot
        tokens[:, -1] = 59
        token_pe = self.token_pos_enc(self.max_dec_len)
        kv_cache = torch.zeros(x.shape[0], self.max_dec_len, self.dec_dim, self.num_dec_layers, 2,
                               dtype=x.dtype, device=x.device)
        for idx in range(self.max_dec_len - 2):
            if torch.all(torch.any(tokens[:, :idx + 1] == 59, dim=1)):
                break
            self.dec.infer_step(enc_out, tokens, token_pe, pad_mask, idx, kv_cache)
        return tokens[:, 1:]

    def slow_infer(self, x, sot):
        pad_mask = torch.all(torch.all(x == 0, dim=3), dim=2)  # [N, L]
        enc_out = self.enc(x, pad_mask)
        tokens = torch.full((x.shape[0], self.max_dec_len), fill_value=61, dtype=torch.long, device=x.device)
        if sot == 'train':
            sot = 60
        if sot == 'supp':
            sot = 61
        tokens[:, 0] = sot
        tokens[:, -1] = 59
        token_pe = self.token_pos_enc(self.max_dec_len)
        for idx in range(self.max_dec_len - 2):  # TODO can shorten for loop
            if torch.all(torch.any(tokens[:, :idx + 1] == 59, dim=1)):
                break
            dec_out = self.dec(enc_out, tokens, token_pe, pad_mask)
            tokens[:, idx + 1] = torch.argmax(dec_out[:, idx], dim=-1)
        return tokens[:, 1:]
