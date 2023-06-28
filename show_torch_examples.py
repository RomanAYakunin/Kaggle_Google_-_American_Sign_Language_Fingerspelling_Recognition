import os
import torch
from torch.utils.data import TensorDataset
from model import Model
from training import train
from sklearn.utils import shuffle
from torchinfo import summary
from dataset import FeatureGenerator, NPZDataset, get_dataloader
from utils import load_arrs, save_arrs, get_paths, train_val_split
import numpy as np
from utils import get_seq_ids, train_val_split, get_phrases, label_to_phrase
from dataset import get_seqs
import editdistance

_, val_seq_ids = train_val_split()
seq_id = shuffle(val_seq_ids)[0]
seq = get_seqs([seq_id])[0]

model = Model().cuda()
model.load_state_dict(torch.load('saved_models/test_model.pt'))
model.eval()

with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model.infer(torch.from_numpy(seq).unsqueeze(0).cuda()).squeeze(0)
    output = output[:torch.argwhere(output == 59).ravel()[0]]
    model_phrase = label_to_phrase(output)

phrase = get_phrases([seq_id])[0]
print(phrase)
print(model_phrase)
edit_dist = editdistance.eval(phrase, model_phrase)
print('phrase len:', len(phrase))
print('model phrase len:', len(model_phrase))
print('edit dist:', edit_dist)
print('acc score:', (len(phrase) - edit_dist)/len(phrase))
