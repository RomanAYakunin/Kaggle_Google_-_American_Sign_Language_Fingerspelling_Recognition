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

tflite_model_path = 'submissions/model.tflite'
print(f'model size: {os.path.getsize(tflite_model_path) / 2**20} MB')

interpreter = tf.lite.Interpreter(tflite_model_path)

prediction_fn = interpreter.get_signature_runner("serving_default")

_, val_seq_ids = train_val_split(shuffle(get_seq_ids(), random_state=9773)[:60000])
seqs = get_seqs(val_seq_ids)
labels = phrases_to_labels(get_phrases(val_seq_ids))

time_sum, len_sum, dist_sum = 0, 0, 0  # TODO test with torch model
for i, (seq, label) in enumerate(pbar := tqdm(list(zip(seqs, labels)), file=sys.stdout)):
    pbar.set_description(f'validating tflite model')
    len_sum += len(label)
    seq = seq.transpose(0, 2, 1).reshape((seq.shape[0], -1)).astype(np.float32)
    time_start = time.time()
    output = prediction_fn(inputs=seq)
    time_sum += time.time() - time_start
    output = np.argmax(output['outputs'], axis=-1)
    dist_sum += editdistance.eval(output, label - 1)
    pbar.set_postfix_str(f'mean accuracy = {(len_sum - dist_sum) / len_sum:.9f}, '
                         f'mean pred time = {1e3 * time_sum / (i + 1):.9f} ms')
print('accuracy:', (len_sum - dist_sum) / len_sum)
print('pred time:', 1e3 * time_sum / len(seqs), 'ms')




