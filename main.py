import torch
from torchinfo import summary

from dataset import FeatureGenerator, get_dataloader
from torch_model.model import Model
from training import train

# train_seq_ids, val_seq_ids = train_val_split()
# NPZDataset.create(train_seq_ids, 'proc_data/train.npz')
# NPZDataset.create(val_seq_ids, 'proc_data/val.npz')

batch_size = 64
model = Model().cuda()
model.load_state_dict(torch.load('saved_models/train_best_model.pt'))
model.eval()  # EVALING!
FG = FeatureGenerator()
summary(model, input_size=[(2, 170, FG.num_points, FG.num_axes), (2, 20)], dtypes=[torch.float32, torch.long])
model.train()  # TRAINING!

train_dataloader = get_dataloader(save_path='proc_data/train.npz', batch_size=batch_size, shuffle=True)
val_dataloader = get_dataloader(save_path='proc_data/val.npz', batch_size=batch_size, shuffle=False)

EPOCHS = 80
SWA_EPOCHS = 40
WARMDOWN_EPOCHS = 40
WARMDOWN_FACTOR = 0.1

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)

train(model, optimizer, train_dataloader, epochs=EPOCHS, swa_epochs=SWA_EPOCHS, warmdown_epochs=WARMDOWN_EPOCHS,
      warmdown_factor=WARMDOWN_FACTOR, val_dataloader=val_dataloader)
