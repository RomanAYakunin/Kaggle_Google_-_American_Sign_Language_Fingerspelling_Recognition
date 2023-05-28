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
from utils import get_seq_ids, train_val_split

# train_seq_ids, val_seq_ids = train_val_split()
# NPZDataset.create(train_seq_ids, 'proc_data/train.npz', crop_labels=True)
# NPZDataset.create(val_seq_ids, 'proc_data/val.npz', crop_labels=False)

batch_size = 128
model = Model().cuda()
FG = FeatureGenerator()
summary(model, input_size=(2, FG.max_len, FG.num_points, FG.num_axes))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)  # TODO restart with proper lr

train_dataloader = get_dataloader(save_path='proc_data/train.npz', batch_size=batch_size, shuffle=True)
val_dataloader = get_dataloader(save_path='proc_data/val.npz', batch_size=batch_size, shuffle=False)

train(model, train_dataloader, epochs=1000, optimizer=optimizer, val_dataloader=val_dataloader,
      save_path='saved_models/test_model.pt')