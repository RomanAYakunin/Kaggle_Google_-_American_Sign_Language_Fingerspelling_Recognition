import torch
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
onnx_model_path = 'onnx_conversion/model.onnx'
tf_model_dir = 'onnx_conversion/tf_model'
tf_infer_model_dir = 'onnx_conversion/tf_infer_model'
tflite_infer_model_path = 'submissions/model.tflite'

model = Model()
model.load_state_dict(torch.load(torch_model_path))
model.eval()
model_inputs = torch.ones((1, 40, FG.num_points, FG.num_axes))

torch.onnx.export(
    model,                    # PyTorch Model
    model_inputs,             # Input tensor
    onnx_model_path,          # Output file (eg. 'output_model.onnx')
    opset_version=12,         # Operator support version
    input_names=['x'],        # Input tensor name (arbitrary)
    output_names=['output'],  # Output tensor name (arbitrary)
    dynamic_axes={
        'x': {1: 'seq_len'},
    }
)

onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_dir)


class InferenceModel(tf.Module):
    def __init__(self):
        super(InferenceModel, self).__init__()
        self.range_arr = tf.range(FG.max_len, dtype=tf.int64)
        self.model = tf.saved_model.load(tf_model_dir)
        self.model.trainable = False

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, len(columns)], dtype=tf.float32, name='inputs')
    ])
    def call(self, x):
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)[:FG.max_len]
        x = tf.transpose(tf.reshape(x, (1, -1, FG.num_axes, FG.num_points)), (0, 1, 3, 2))
        output = self.model(**{'x': x})['output'][0, :]
        output = tf.argmax(output, axis=1)
        shifted_output = tf.concat([tf.zeros((1,), dtype=tf.int64), output[:-1]], axis=0)
        output = output[tf.math.logical_and(tf.math.not_equal(output, shifted_output),
                                            tf.math.not_equal(output, tf.zeros_like(output)))] - 1
        output = tf.cond(tf.shape(output)[0] == 0, lambda: tf.zeros((1,), dtype=tf.int64), lambda: output)
        idxs = tf.stack([self.range_arr[:tf.shape(output)[0]], output], axis=1)
        one_hot = tf.scatter_nd(indices=idxs, updates=tf.ones_like(output, dtype=tf.float32),
                                shape=(tf.shape(output)[0], 59))
        output_tensors = {}
        output_tensors['outputs'] = one_hot
        return output_tensors

infer_model = InferenceModel()
tf.saved_model.save(infer_model, tf_infer_model_dir, signatures={'serving_default': infer_model.call})
converter = tf.lite.TFLiteConverter.from_saved_model(tf_infer_model_dir)
tflite_model = converter.convert()

with open(tflite_infer_model_path, 'wb') as file:
    file.write(tflite_model)

# import os
# import json
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from utils import get_seq_ids, train_val_split, get_phrases, phrases_to_labels
# from sklearn.utils import shuffle
# from tqdm import tqdm
# import sys
# import editdistance
# import time
# from dataset import get_seqs
# from copy import deepcopy
#
# tflite_model_path = 'submissions/model.tflite'
# print(f'model size: {os.path.getsize(tflite_model_path) / 2**20} MB')
#
# interpreter = tf.lite.Interpreter(tflite_model_path)
#
# prediction_fn = interpreter.get_signature_runner("serving_default")
#
# _, val_seq_ids = train_val_split(shuffle(get_seq_ids(), random_state=9773)[:60000])
# val_seq_ids = val_seq_ids[:10]
# seqs = get_seqs(val_seq_ids)
# labels = phrases_to_labels(get_phrases(val_seq_ids))
#
# time_sum, len_sum, dist_sum = 0, 0, 0
# for i, (seq, label) in enumerate(pbar := tqdm(list(zip(seqs, labels)), file=sys.stdout)):
#     pbar.set_description(f'validating tflite model')
#     len_sum += len(label)
#     orig_seq = deepcopy(seq)
#     seq = seq.transpose(0, 2, 1).reshape((seq.shape[0], -1)).astype(np.float32)
#     time_start = time.time()
#     # output = prediction_fn(inputs=seq)
#     # output = infer_model.call(tf.convert_to_tensor(seq))
#
#     range_arr = tf.range(FG.max_len, dtype=tf.int64)
#     model = tf.saved_model.load(tf_model_dir)
#
#     x = seq
#     x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)[:FG.max_len]
#     x = tf.transpose(tf.reshape(x, (1, -1, FG.num_axes, FG.num_points)), (0, 1, 3, 2))
#     output = model(**{'x': x})['output'][0, :]
#     one_hot = tf.zeros_like(output)
#     output = tf.argmax(output, axis=1)
#     shifted_output = tf.concat([tf.zeros((1,), dtype=tf.int64), output[:-1]], axis=0)
#     output = output[tf.math.logical_and(tf.math.not_equal(output, shifted_output),
#                                         tf.math.not_equal(output, tf.zeros_like(output)))] - 1
#     idxs = tf.stack([range_arr[:output.shape[0]], output], axis=1)
#     one_hot = one_hot[:idxs.shape[0], :59]
#     one_hot = tf.tensor_scatter_nd_update(one_hot, indices=idxs, updates=tf.ones_like(output, dtype=tf.float32))
#     output = one_hot
#
#     time_sum += time.time() - time_start
#     output = np.argmax(output, axis=-1)
#     dist_sum += editdistance.eval(output, label)
#     pbar.set_postfix_str(f'mean accuracy = {(len_sum - dist_sum) / len_sum:.9f}, '
#                          f'mean pred time = {1e3 * time_sum / (i + 1):.9f} ms')
# print('accuracy:', (len_sum - dist_sum) / len_sum)
# print('pred time:', 1e3 * time_sum / len(seqs), 'ms')