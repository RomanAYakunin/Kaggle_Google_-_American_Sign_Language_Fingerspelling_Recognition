import sys
import time

import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

from dataset import get_seqs
from tf_decoder import MultiHeadAttention, Decoder
from tf_encoder import PositionalEncoding, SlidingATTN, AxisLayerNorm, Encoder
from tf_model.tf_modules import Linear, LayerNorm, ELU, AutoModule
from torch_model.model import Model
from utils import train_val_split


def compare_outputs(module, tf_module, inputs, eps=1e-5):
    for x in inputs:
        torch_x = torch.from_numpy(x) if type(x) is np.ndarray else x
        with torch.no_grad():
            out = module(torch_x).numpy()
        tf_out = tf_module(x).numpy()
        if out.shape != tf_out.shape:
            raise RuntimeError("output shapes don't match")
        print('output shape:', out.shape)
        diff = np.abs(out - tf_out).max()
        print('diff:', diff)
        if diff > eps:
            raise RuntimeError('output diff exceeds eps')


def linear_test():
    module = model.enc.input_net[0]
    tf_module = Linear(module)
    x = np.random.normal(size=(11, module.weight.detach().numpy().shape[1])).astype(np.float32)
    compare_outputs(module, tf_module, [x])
    print('LINEAR TEST PASSED\n')


def layernorm_test():
    module = model.enc.input_net[1]
    tf_module = LayerNorm(module)
    x = np.random.normal(size=(11, module.weight.detach().numpy().shape[0])).astype(np.float32)
    compare_outputs(module, tf_module, [x])
    print('LAYERNORM TEST PASSED\n')


def elu_test():
    module = model.enc.input_net[2]
    tf_module = ELU()
    x = np.random.normal(size=(11, 1000)).astype(np.float32)
    compare_outputs(module, tf_module, [x])
    print('ELU TEST PASSED\n')


def automodule_test():
    module = model.enc.input_net
    tf_module = AutoModule(module)
    x = np.random.normal(size=(11, 100, model.enc.input_dim)).astype(np.float32)
    compare_outputs(module, tf_module, [x])
    print('AUTOMODULE TEST PASSED\n')


def pos_enc_test():
    module = model.enc.pos_enc
    tf_module = PositionalEncoding(module)
    x = 100
    compare_outputs(module, tf_module, [x])
    print('POSITIONAL ENCODING TEST PASSED\n')


def sliding_attn_test():
    module = model.enc.sliding_attn1
    tf_module = SlidingATTN(module)
    x = np.random.normal(size=(11, 100, model.enc.dim)).astype(np.float32)
    mask = np.zeros(shape=(11, 100), dtype=bool)
    with torch.no_grad():
        out = module(torch.from_numpy(x), torch.from_numpy(mask)).numpy()
    tf_out = tf_module(x).numpy()
    if out.shape != tf_out.shape:
        raise RuntimeError("output shapes don't match")
    print('output shape:', out.shape)
    diff = np.abs(out - tf_out).max()
    print('diff:', diff)
    eps = 1e-5
    if diff > eps:
        raise RuntimeError('output diff exceeds eps')
    print('SLIDING ATTN TEST PASSED\n')


def x_norm_test():
    module = model.enc.x_norm
    tf_module = AxisLayerNorm(module)
    x = np.random.normal(size=(11, 100, module.num_points, module.num_axes)).astype(np.float32)
    compare_outputs(module, tf_module, [x])
    print('X NORM TEST PASSED\n')


def feature_norm_test():
    module = model.enc.feature_norms[0]
    tf_module = AxisLayerNorm(module)
    x = np.random.normal(size=(11, 100, module.num_points, module.num_axes)).astype(np.float32)
    compare_outputs(module, tf_module, [x])
    print('FEATURE NORM TEST PASSED\n')


def encoder_test():
    module = model.enc
    tf_module = Encoder(module)
    _, val_seq_ids = train_val_split()
    val_seq_ids = val_seq_ids[:100]
    seqs = get_seqs(val_seq_ids)
    max_diff = 0
    torch_time_sum = 0
    tf_time_sum = 0
    for x in tqdm(seqs, file=sys.stdout):
        x = np.expand_dims(x, 0).astype(np.float32)
        torch_x = torch.from_numpy(x) if type(x) is np.ndarray else x
        with torch.no_grad():
            time_start = time.time()
            out = module(torch_x, mask=torch.all(torch.all(torch_x == 0, dim=3), dim=2)).numpy()
            torch_time_sum += time.time() - time_start
        time_start = time.time()
        tf_out = tf_module(x).numpy()
        tf_time_sum += time.time() - time_start
        if out.shape != tf_out.shape:
            raise RuntimeError("output shapes don't match")
        max_diff = max(max_diff, np.abs(out - tf_out).max())
    print('max diff:', max_diff)
    print('mean torch time:', torch_time_sum / len(seqs))
    print('mean tf time:', tf_time_sum / len(seqs))
    print('INSPECT DIFF MANUALLY FOR ENCODER TEST\n')


def mha_test():
    module = model.dec.layers[0].self_attn.mha
    tf_module = MultiHeadAttention(module)
    q = np.random.normal(size=(11, 40, module.dim)).astype(np.float32)
    k = np.random.normal(size=(11, 100, module.dim)).astype(np.float32)
    v = np.random.normal(size=(11, 100, module.dim)).astype(np.float32)
    torch_mask = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    with torch.no_grad():
        out = module(
            torch.from_numpy(q),
            torch.from_numpy(k),
            torch.from_numpy(v),
            torch_mask
        ).numpy()
    tf_out = tf_module(q, k, v).numpy()
    if out.shape != tf_out.shape:
        raise RuntimeError("output shapes don't match")
    diff = np.abs(out - tf_out).max()
    print('diff:', diff)
    print('INSPECT DIFF MANUALLY FOR MHA TEST\n')


def self_attn_test():
    module = model.dec.layers[0].self_attn
    tf_module = Decoder(module)



model = Model()
model.load_state_dict(torch.load('../saved_models/test_model.pt'))
model.eval()

tf.config.set_visible_devices([], 'GPU')

linear_test()
layernorm_test()
elu_test()
automodule_test()
pos_enc_test()
sliding_attn_test()
x_norm_test()
feature_norm_test()
encoder_test()
mha_test()
