import torch
from torchinfo import summary

from dataset import FeatureGenerator, get_dataloader
from torch_model.model import Model
from training import train

# train_seq_ids, val_seq_ids = train_val_split()
# NPZDataset.create(train_seq_ids, 'proc_data/train.npz')
# NPZDataset.create(val_seq_ids, 'proc_data/val.npz')

batch_size = 64
model = Model().cuda()  # TODO use or not use checkpoints
model.load_state_dict(torch.load('saved_models/train_best_model.pt'))  # TODO change to base model if needed
model.eval()  # EVALING!
FG = FeatureGenerator()
summary(model, input_size=[(2, 170, FG.num_points, FG.num_axes), (2, 20)], dtypes=[torch.float32, torch.long])
model.train()  # TRAINING!
# TODO try smaller decoder


# TODO SCALE DOWN LEARNING RATE!!!!
# TODO can make wd bigger to compensate for lack of dropout

train_dataloader = get_dataloader(save_path='proc_data/train.npz', batch_size=batch_size, shuffle=True)
val_dataloader = get_dataloader(save_path='proc_data/val.npz', batch_size=batch_size, shuffle=False)

EPOCHS = 81
SWA_EPOCHS = 80
WARMDOWN_EPOCHS = 1
WARMDOWN_FACTOR = 1

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)  # TODO try ranger21

# optimizer = Ranger21(model.parameters(),
#                      lr=1e-3, weight_decay=0.1,
#                      lookahead_mergetime=10,
#                      num_epochs=EPOCHS,
#                      num_batches_per_epoch=len(train_dataloader),
#                      warmdown_active=False,
#                      logging_active=False)

train(model, optimizer, train_dataloader, epochs=EPOCHS, swa_epochs=SWA_EPOCHS, warmdown_epochs=WARMDOWN_EPOCHS,
      warmdown_factor=WARMDOWN_FACTOR, val_dataloader=val_dataloader)
