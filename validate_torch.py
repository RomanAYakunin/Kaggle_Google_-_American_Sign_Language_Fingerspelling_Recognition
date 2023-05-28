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
from model import Model
import torch
from utils import proc_model_output

model = Model()
model.load_state_dict(torch.load('saved_models/debug_model.pt'))
model.eval()

_, val_seq_ids = train_val_split()
seqs = get_seqs(val_seq_ids)
labels = phrases_to_labels(get_phrases(val_seq_ids))

time_sum, len_sum, dist_sum = 0, 0, 0  # TODO test with torch model
for i, (seq, label) in enumerate(pbar := tqdm(list(zip(seqs, labels)), file=sys.stdout)):
    pbar.set_description(f'validating tflite model')
    len_sum += len(label)
    seq = seq.astype(np.float32)
    time_start = time.time()
    output = model(torch.from_numpy(seq).unsqueeze(0)).squeeze(0).detach()
    output = torch.argmax(output, dim=-1)
    output = proc_model_output(output).numpy()
    time_sum += time.time() - time_start
    dist_sum += editdistance.eval(output.tolist(), label.tolist())
    pbar.set_postfix_str(f'mean accuracy = {(len_sum - dist_sum) / len_sum:.9f}, '
                         f'mean pred time = {1e3 * time_sum / (i + 1):.9f} ms')
print('accuracy:', (len_sum - dist_sum) / len_sum)
print('pred time:', 1e3 * time_sum / len(seqs), 'ms')




