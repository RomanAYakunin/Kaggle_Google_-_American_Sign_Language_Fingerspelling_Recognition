import math
import torch
import torch.nn as nn
from dataset import FeatureGenerator
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
from encoder import Encoder
from decoder import Decoder


class Model(nn.Module):
    def __init__(self, use_checkpoints=True):
        super(Model, self).__init__()
        self.num_dec_layers = 2
        self.dec_dim = 256
        self.num_dec_heads = 16
        self.max_dec_len = 50
        self.enc = Encoder(self.num_dec_layers, self.dec_dim, use_checkpoints)
        self.dec = Decoder(self.num_dec_layers, self.dec_dim, self.num_dec_heads, self.max_dec_len)

    def forward(self, x, tokens):
        pad_mask = torch.all(torch.all(x == 0, dim=3), dim=2)  # [N, L]
        enc_out = self.enc(x, pad_mask)  # TODO move the K, V lin to here
        dec_out = self.dec(enc_out, tokens, pad_mask)
        return dec_out

    def infer(self, x):  # TODO make smarter inference
        pad_mask = torch.all(torch.all(x == 0, dim=3), dim=2)  # [N, L]
        enc_out = self.enc(x, pad_mask)
        tokens = torch.full((x.shape[0], self.max_dec_len), fill_value=61, dtype=torch.long, device=x.device)
        tokens[:, 0] = 60
        tokens[:, -1] = 59
        for i in range(self.max_dec_len - 2):  # TODO can shorten for loop
            dec_out = self.dec(enc_out, tokens, pad_mask)
            tokens[:, i + 1] = torch.argmax(dec_out[:, i], dim=-1)
        return tokens[:, 1:]
