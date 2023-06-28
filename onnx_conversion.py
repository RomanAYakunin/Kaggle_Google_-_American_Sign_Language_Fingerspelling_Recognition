import torch
import torch.nn as nn
import json
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
from model import Model
from dataset import FeatureGenerator, get_column_names

FG = FeatureGenerator()
columns = get_column_names().tolist()

column_dict = {"selected_columns": columns}

with open(f'submissions/inference_args.json', 'w') as file:
    json.dump(column_dict, file)

torch_model_path = 'saved_models/test_model.pt'
onnx_enc_path = 'onnx_conversion/enc.onnx'
onnx_pos_enc_path = 'onnx_conversion/pos_enc.onnx'
onnx_dec_path = 'onnx_conversion/dec.onnx'
tf_enc_dir = 'onnx_conversion/tf_enc'
tf_pos_enc_dir = 'onnx_conversion/tf_pos_enc'
tf_dec_dir = 'onnx_conversion/tf_dec'
tf_infer_model_dir = 'onnx_conversion/tf_infer_model'
tflite_infer_model_path = 'submissions/model.tflite'

model = Model(use_checkpoints=False)
model.load_state_dict(torch.load(torch_model_path))
model.eval()

enc = model.enc
enc.eval()
enc_inputs = (
    torch.zeros((1, 40, FG.num_points, FG.num_axes), dtype=torch.float32),
)

torch.onnx.export(
    enc,                    # PyTorch Model
    enc_inputs,             # Input tensor
    onnx_enc_path,          # Output file (eg. 'output_model.onnx')
    opset_version=12,         # Operator support version
    input_names=['x'],        # Input tensor name (arbitrary)
    output_names=['output'],  # Output tensor name (arbitrary)
    dynamic_axes={
        'x': {0: 'batch_size', 1: 'len'},
        # 'output': {1: 'len'}
    }
)

onnx_enc = onnx.load(onnx_enc_path)
tf_rep = prepare(onnx_enc)
tf_rep.export_graph(tf_enc_dir)

# =============================================================================
#
# pos_enc = model.token_pos_enc
# pos_enc.eval()
# pos_enc_inputs = model.max_dec_len
#
# torch.onnx.export(
#     pos_enc,                    # PyTorch Model
#     pos_enc_inputs,             # Input tensor
#     onnx_pos_enc_path,          # Output file (eg. 'output_model.onnx')
#     opset_version=12,         # Operator support version
#     input_names=['x_len'],        # Input tensor name (arbitrary)
#     output_names=['output'],  # Output tensor name (arbitrary)
# )
#
# onnx_pos_enc = onnx.load(onnx_pos_enc_path)
# tf_rep = prepare(onnx_pos_enc)
# tf_rep.export_graph(tf_pos_enc_dir)
#
# # =============================================================================
#
#
# class InferWrapper(nn.Module):
#     def __init__(self, module):
#         super(InferWrapper, self).__init__()
#         self.module = module
#
#     def forward(self, *args):
#         return self.module.infer_step(*args)  # TODO check if works
#
#
# dec = InferWrapper(model.dec)
# dec.eval()
# dec_inputs = (
#     torch.zeros((1, 40, model.dec_dim, model.num_dec_layers, 2), dtype=torch.float32),
#     torch.zeros((1, model.max_dec_len), dtype=torch.int64),
#     torch.zeros((1, model.max_dec_len, model.dec_dim), dtype=torch.float32),
#     torch.zeros((1, 40), dtype=torch.bool),
#     0,
#     torch.zeros((1, model.max_dec_len, model.dec_dim, model.num_dec_layers, 2))
# )
#
# torch.onnx.export(
#     dec,                    # PyTorch Model
#     dec_inputs,             # Input tensor
#     onnx_dec_path,          # Output file (eg. 'output_model.onnx')
#     opset_version=12,         # Operator support version
#     input_names=[  # Input tensor name (arbitrary)
#         'enc_out',
#         'tokens',
#         'token_pe',
#         'pad_mask',
#         'idx',
#         'kv_cache'
#     ],
#     output_names=[
#         'out_enc_out',
#         'out_tokens',
#         'out_token_pe',
#         'out_pad_mask',
#         'out_idx',
#         'out_kv_cache'
#     ],  # Output tensor name (arbitrary)
#     dynamic_axes={
#         'enc_out': [1],
#         'pad_mask': [1],
#         'out_enc_out': [1],
#         'out_pad_mask': [1]
#     }
# )
#
# onnx_dec = onnx.load(onnx_dec_path)
# tf_rep = prepare(onnx_dec)
# tf_rep.export_graph(tf_dec_dir)
#
# # =============================================================================


class InferenceModel(tf.Module):
    def __init__(self):
        super(InferenceModel, self).__init__()
        self.num_dec_layers = model.num_dec_layers
        self.dec_dim = model.dec_dim
        self.num_dec_heads = model.num_dec_heads
        self.max_dec_len = model.max_dec_len  # TODO unbind this later?
        self.enc = tf.saved_model.load(tf_enc_dir)
        self.pos_enc = tf.saved_model.load(tf_pos_enc_dir)
        self.dec = tf.saved_model.load(tf_dec_dir)
        self.enc.trainable = False
        self.pos_enc.trainable = False
        self.dec.trainable = False

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, len(columns)], dtype=tf.float32, name='inputs')
    ])
    def call(self, x):
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)[:FG.max_len]
        x = tf.transpose(tf.reshape(x, (1, -1, FG.num_axes, FG.num_points)), (0, 1, 3, 2))

        pad_mask = tf.cast(tf.zeros((1, tf.shape(x)[1])), dtype=tf.bool)
        enc_out = self.enc(**{'x': x})['output']  # TODO redo mask if needed
        print(enc_out.shape)
        tokens = tf.cast(tf.fill([1, self.max_dec_len - 2], value=61), dtype=tf.int64)
        tokens = tf.concat([tf.cast(tf.fill((1, 1), value=60), dtype=tf.int64),
                            tokens,
                            tf.cast(tf.fill((1, 1), value=59), dtype=tf.int64)], axis=1)
        token_pe = self.pos_enc(**{'x_len': self.max_dec_len})['output']  # TODO may have to be tensor
        kv_cache = tf.zeros([x.shape[0], self.max_dec_len, self.dec_dim, self.num_dec_layers, 2], dtype=tf.float32)
        loop_vars = [enc_out, tokens, token_pe, pad_mask, tf.cast(0, dtype=tf.int64),
                     kv_cache]  # TODO idx may have to be tensor
        cond = lambda enc_out, tokens, token_pe, pad_mask, idx, kv_cache: tokens[0, idx] != 59
        loop_vars = tf.while_loop(cond, self.body, loop_vars)
        # loop_vars = self.body(*loop_vars)
        tokens = loop_vars[1]
        tokens = tokens[:tf.reshape(tf.where(tokens == 59), [-1])[0]]
        output = tf.one_hot(tokens, depth=59)
        output_tensors = {}
        output_tensors['outputs'] = output
        return output_tensors

    def body(self, enc_out, tokens, token_pe, pad_mask, idx, kv_cache):  # TODO try in place
        dec_out = self.dec(**{'enc_out': enc_out,
                              'tokens': tokens,
                              'token_pe': token_pe,
                              'pad_mask': pad_mask,
                              'idx': idx,
                              'kv_cache': kv_cache})
        return dec_out['out_enc_out'], dec_out['out_tokens'], dec_out['out_token_pe'], dec_out['out_pad_mask'], \
               idx + 1, dec_out['out_kv_cache']


infer_model = InferenceModel()
tf.saved_model.save(infer_model, tf_infer_model_dir, signatures={'serving_default': infer_model.call})
converter = tf.lite.TFLiteConverter.from_saved_model(tf_infer_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # TODO look at the experimental stuff you saw
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open(tflite_infer_model_path, 'wb') as file:
    file.write(tflite_model)

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import get_seq_ids, train_val_split, get_phrases, phrases_to_labels
from sklearn.utils import shuffle
from tqdm import tqdm
import sys
import editdistance
import time
from dataset import get_seqs
from copy import deepcopy
#
# tflite_model_path = 'submissions/model.tflite'
# print(f'model size: {os.path.getsize(tflite_model_path) / 2**20} MB')
#
# interpreter = tf.lite.Interpreter(tflite_model_path)
#
# prediction_fn = interpreter.get_signature_runner("serving_default")
#
_, val_seq_ids = train_val_split(shuffle(get_seq_ids(), random_state=9773)[:60000])
val_seq_ids = val_seq_ids[:5]
seqs = get_seqs(val_seq_ids)
labels = phrases_to_labels(get_phrases(val_seq_ids))
#
time_sum, len_sum, dist_sum = 0, 0, 0
tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)
for i, (seq, label) in enumerate(pbar := tqdm(list(zip(seqs, labels)), file=sys.stdout)):
    pbar.set_description(f'validating tflite model')
    len_sum += len(label)
    orig_seq = deepcopy(seq)
    seq = seq.transpose(0, 2, 1).reshape((seq.shape[0], -1)).astype(np.float32)
    time_start = time.time()
    # output = prediction_fn(inputs=seq)
    # output = infer_model.call(tf.convert_to_tensor(seq))

    output = infer_model.call(seq)['outputs']

    time_sum += time.time() - time_start
    output = np.argmax(output, axis=-1)
    dist_sum += editdistance.eval(output.tolist(), label.tolist())
    pbar.set_postfix_str(f'mean accuracy = {(len_sum - dist_sum) / len_sum:.9f}, '
                         f'mean pred time = {1e3 * time_sum / (i + 1):.9f} ms')
print('accuracy:', (len_sum - dist_sum) / len_sum)
print('pred time:', 1e3 * time_sum / len(seqs), 'ms')
