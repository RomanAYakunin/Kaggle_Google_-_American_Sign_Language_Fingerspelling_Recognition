import torch
from torch_model.model import Model
from training import train
from torchinfo import summary
from dataset import FeatureGenerator, get_dataloader, NPZDataset
from utils import get_seq_ids

# train_seq_ids = get_seq_ids()
# NPZDataset.create(train_seq_ids, 'proc_data/all.npz')

batch_size = 64
model = Model().cuda()
model.load_state_dict(torch.load('saved_models/train_last_model.pt'))
model.eval()  # EVALING!
FG = FeatureGenerator()
summary(model, input_size=[(2, 170, FG.num_points, FG.num_axes), (2, 20)], dtypes=[torch.float32, torch.long])
model.train()  # TRAINING!

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)

train_dataloader = get_dataloader(save_path='proc_data/all.npz', batch_size=batch_size, shuffle=True)

train(model, optimizer, train_dataloader, epochs=80, swa_epochs=40, warmdown_epochs=40, warmdown_factor=0.1)
