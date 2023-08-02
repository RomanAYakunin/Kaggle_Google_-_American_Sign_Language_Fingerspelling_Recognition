import torch
from torch_model.model import Model
from training import train
from torchinfo import summary
from dataset import FeatureGenerator, get_dataloader, NPZDataset

# train_seq_ids, val_seq_ids = train_val_split()
# NPZDataset.create(train_seq_ids, 'proc_data/train.npz')
# NPZDataset.create(val_seq_ids, 'proc_data/val.npz')

batch_size = 64
model = Model().cuda()  # TODO use or not use checkpoints
# model.load_state_dict(torch.load('saved_models/best_model.pt'))  # TODO change to base model if needed
model.eval()
FG = FeatureGenerator()
summary(model, input_size=[(2, 170, FG.num_points, FG.num_axes), (2, 20)], dtypes=[torch.float32, torch.long])
model.train()
# TODO try smaller decoder
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)  # TODO make wd=0.1

train_dataloader = get_dataloader(save_path='proc_data/train.npz', batch_size=batch_size, shuffle=True)
val_dataloader = get_dataloader(save_path='proc_data/val.npz', batch_size=batch_size, shuffle=False)

train(model, optimizer, train_dataloader, epochs=80, swa_epochs=20, warmdown_epochs=20, warmdown_factor=0.1,
      val_dataloader=val_dataloader)
