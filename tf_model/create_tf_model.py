import torch
import json
from pathlib import Path
from torch_model.model import Model
from dataset import FeatureGenerator, get_column_names
import tensorflow as tf
from tf_encoder import PositionalEncoding, Encoder
from tf_decoder import Decoder
from torch_model.model import Model

PROJECT_DIR = str(Path(__file__).parents[1])

torch_model_path = f'{PROJECT_DIR}/saved_models/train_best_model.pt'
tf_infer_model_dir = f'{PROJECT_DIR}/conversion/tf_infer_model'
tflite_infer_model_path = f'{PROJECT_DIR}/submissions/model.tflite'

FG = FeatureGenerator()
columns = get_column_names().tolist()

column_dict = {"selected_columns": columns}

with open(f'{PROJECT_DIR}/submissions/inference_args.json', 'w') as file:
    json.dump(column_dict, file)


class InferenceModel(tf.Module):
    def __init__(self, module):
        super(InferenceModel, self).__init__()
        self.num_dec_layers = module.num_dec_layers
        self.dec_dim = module.dec_dim
        self.num_dec_heads = module.num_dec_heads
        self.max_dec_len = module.max_dec_len
        self.enc = Encoder(module.enc)
        self.token_pos_enc = PositionalEncoding(module.token_pos_enc)
        self.dec = Decoder(module.dec)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, len(columns)], dtype=tf.float32, name='inputs')
    ])
    def call(self, x):
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)[:FG.aug_max_len]
        x = tf.transpose(tf.reshape(x, (1, -1, FG.num_axes, FG.num_points)), (0, 1, 3, 2))

        enc_out = self.enc(x)
        tokens = tf.cast(tf.fill((1, 1), value=60), tf.int64)
        token_pe = self.token_pos_enc(self.max_dec_len)
        kv_cache = tf.zeros((1, 0, self.dec_dim, self.num_dec_layers, 2), dtype=tf.float32)
        idx = tf.cast(0, dtype=tf.int64)
        loop_vars = [enc_out, tokens, token_pe, kv_cache, idx]
        shape_invariants = [
            enc_out.get_shape(),
            tf.TensorShape([1, None]),
            token_pe.get_shape(),
            tf.TensorShape([1, None, self.dec_dim, self.num_dec_layers, 2]),
            idx.get_shape()
        ]
        cond = lambda enc_out, tokens, token_pe, kv_cache, idx: \
            tf.logical_and(tokens[0, -1] != 59, idx < self.max_dec_len - 2)
        loop_vars = tf.while_loop(cond, self.dec, loop_vars, shape_invariants=shape_invariants)
        tokens = loop_vars[1][0]
        tokens = tokens[1: tf.reshape(tf.where(tokens == 59), (-1,))[0]]
        tokens = tf.cond(tf.shape(tokens)[0] == 0, lambda: tf.zeros((1,), dtype=tf.int64), lambda: tokens)
        output = tf.one_hot(tokens, depth=59)
        output_tensors = {}
        output_tensors['outputs'] = output
        return output_tensors


torch_model = Model()
torch_model.load_state_dict(torch.load(torch_model_path))
torch_model.eval()

infer_model = InferenceModel(torch_model)
tf.saved_model.save(infer_model, tf_infer_model_dir, signatures={'serving_default': infer_model.call})

converter = tf.lite.TFLiteConverter.from_saved_model(tf_infer_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # TODO look at the experimental stuff you saw
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open(tflite_infer_model_path, 'wb') as file:
    file.write(tflite_model)


# import numpy as np
# import tensorflow as tf
# from utils import get_seq_ids, train_val_split, get_phrases, phrases_to_labels
# from sklearn.utils import shuffle
# from tqdm import tqdm
# import sys
# import editdistance
# import time
# from dataset import get_seqs
# from copy import deepcopy
# import polars as pl
# #
# # tflite_model_path = 'submissions/model.tflite'
# # print(f'model size: {os.path.getsize(tflite_model_path) / 2**20} MB')
# #
# # interpreter = tf.lite.Interpreter(tflite_model_path)
# #
# # prediction_fn = interpreter.get_signature_runner("serving_default")
# #
# train_meta_ids = pl.scan_csv(f'{PROJECT_DIR}/raw_data/train.csv').select('sequence_id').unique().collect().to_numpy().flatten()
#
# _, val_seq_ids = train_val_split()
# # val_seq_ids = val_seq_ids[:100]  # TODO filter out supp seqs
# val_seq_ids = val_seq_ids[np.isin(val_seq_ids, train_meta_ids)]
# val_seq_ids = val_seq_ids[:10]
# seqs = get_seqs(val_seq_ids)
# labels = phrases_to_labels(get_phrases(val_seq_ids))
# #
# time_sum, len_sum, dist_sum = 0, 0, 0
# for i, (seq, label) in enumerate(pbar := tqdm(list(zip(seqs, labels)), file=sys.stdout)):
#     pbar.set_description(f'validating tflite model')
#     len_sum += len(label)
#     orig_seq = deepcopy(seq)
#     seq = seq.transpose(0, 2, 1).reshape((seq.shape[0], -1)).astype(np.float32)
#     time_start = time.time()
#
#     output = infer_model.call(seq)['outputs']
#
#     time_sum += time.time() - time_start
#     output = np.argmax(output, axis=-1)
#     dist_sum += editdistance.eval(output.tolist(), label.tolist())
#     pbar.set_postfix_str(f'mean accuracy = {(len_sum - dist_sum) / len_sum:.9f}, '
#                          f'mean pred time = {1e3 * time_sum / (i + 1):.9f} ms')
# print('accuracy:', (len_sum - dist_sum) / len_sum)
# print('pred time:', 1e3 * time_sum / len(seqs), 'ms')