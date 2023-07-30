import os
import numpy as np
from utils import train_val_split, get_phrases, phrases_to_labels
from tqdm import tqdm
import sys
import editdistance
import time
from dataset import get_seqs
from torch_model.model import Model
import torch
import polars as pl


model_path = 'saved_models/train_swa_model.pt'

print(f'model size: {os.path.getsize(model_path) / 2**20} MB')  # TODO check if maybe 2^ is the problem

model = Model(use_checkpoints=False)
model = torch.optim.swa_utils.AveragedModel(model)
model.load_state_dict(torch.load(model_path))
model = model.module
model.eval()

train_meta_ids = pl.scan_csv('raw_data/train.csv').select('sequence_id').unique().collect().to_numpy().flatten()

_, val_seq_ids = train_val_split()
# val_seq_ids = val_seq_ids[:100]  # TODO filter out supp seqs
val_seq_ids = val_seq_ids[np.isin(val_seq_ids, train_meta_ids)]
# val_seq_ids = val_seq_ids[:10]  # TODO delete
seqs = get_seqs(val_seq_ids)
labels = phrases_to_labels(get_phrases(val_seq_ids))
sot = np.isin(val_seq_ids, train_meta_ids)
sot = ['train' if sot_i else 'supp' for sot_i in sot]

time_sum, len_sum, dist_sum = 0, 0, 0  # TODO test with torch model
for i, (seq, label) in enumerate(pbar := tqdm(list(zip(seqs, labels)), file=sys.stdout)):
    pbar.set_description(f'validating tflite model')
    len_sum += len(label)
    seq = seq.astype(np.float32)
    time_start = time.time()
    with torch.inference_mode():
        # seq = np.concatenate([seq, np.zeros((100, seq.shape[1], seq.shape[2]), dtype=np.float32)])  # TODO remove
        output = model.infer(torch.from_numpy(seq).unsqueeze(0), sot=sot[i]).squeeze(0)
        output = output[:torch.argwhere(output == 59).ravel()[0]]
    time_sum += time.time() - time_start
    # print('\nphrase:', label_to_phrase(label.astype(np.int32)))
    # print('output:', label_to_phrase(output.cpu().numpy().astype(np.int32)))
    dist_sum += editdistance.eval(output.tolist(), label.tolist())
    pbar.set_postfix_str(f'mean accuracy = {(len_sum - dist_sum) / len_sum:.9f}, '
                         f'mean pred time = {1e3 * time_sum / (i + 1):.9f} ms')
print('accuracy:', (len_sum - dist_sum) / len_sum)
print('pred time:', 1e3 * time_sum / len(seqs), 'ms')




