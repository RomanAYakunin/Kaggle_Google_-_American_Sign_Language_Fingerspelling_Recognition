import torch
from torch_model.model import Model
from training import train
from torchinfo import summary
from dataset import FeatureGenerator, get_dataloader, NPZDataset
from utils import get_seq_ids

train_seq_ids = get_seq_ids()
NPZDataset.create(train_seq_ids, 'proc_data/all.npz')

batch_size = 64
model = Model().cuda()
FG = FeatureGenerator()
summary(model, input_size=[(2, 170, FG.num_points, FG.num_axes), (2, 20)], dtypes=[torch.float32, torch.long])

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

train_dataloader = get_dataloader(save_path='proc_data/all.npz', batch_size=batch_size, shuffle=True)

train(model, train_dataloader, epochs=1000, optimizer=optimizer, save_path='saved_models/test_model.pt')
