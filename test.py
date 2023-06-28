import numpy as np
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


# =============================================================================


class InferWrapper(nn.Module):
    def __init__(self, module):
        super(InferWrapper, self).__init__()
        self.module = module

    def forward(self, *args):
        return self.module.infer_step(*args)  # TODO check if works


dec = InferWrapper(model.dec)
dec.eval()
dec_inputs = (
    torch.randn((1, 40, model.dec_dim, model.num_dec_layers, 2), dtype=torch.float32),
    torch.zeros((1, model.max_dec_len), dtype=torch.int64),
    torch.randn((1, model.max_dec_len, model.dec_dim), dtype=torch.float32),
    torch.zeros((1, 40), dtype=torch.bool),
    0,
    torch.randn((1, model.max_dec_len, model.dec_dim, model.num_dec_layers, 2))
)
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

# =============================================================================

tf_dec = tf.saved_model.load(tf_dec_dir)
np_inputs = [0 if isinstance(dec_input, int) else dec_input.numpy() for dec_input in dec_inputs]
call_dict = {
    'enc_out': np_inputs[0],
    'tokens': np_inputs[1],
    'token_pe': np_inputs[2],
    'pad_mask': np_inputs[3],
    'idx': np_inputs[4],
    'kv_cache': np_inputs[5]
}
tf_kv = tf_dec(**call_dict)['out_kv_cache'].numpy()
print(tf_kv)
with torch.no_grad():
    torch_kv = dec(*dec_inputs)[-1].numpy()
print(torch_kv)
print(np.all(tf_kv == torch_kv))
print(np.max(np.abs(tf_kv - torch_kv)))
