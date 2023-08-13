from utils import get_gislr_paths, get_signs
from dataset import get_gislr_data
from sklearn.utils import shuffle
from utils import train_val_split
from dataset import NPZDataset
import torch
from torch_model.model import Model
from training import train
from torchinfo import summary
from dataset import FeatureGenerator, get_dataloader, NPZDataset
from ranger21 import Ranger21

# paths = get_gislr_paths()
# # print(paths)
# # x_list, y_list, xlen_list, ylen_list = get_gislr_data(paths)
# # print(x_list, y_list, xlen_list, ylen_list)
#
# train_seq_ids, val_seq_ids = train_val_split()
# NPZDataset.create(train_seq_ids, 'proc_data/test_train.npz', gislr_paths=paths)
# NPZDataset.create(val_seq_ids, 'proc_data/test_val.npz')

batch_size = 64
model = Model().cuda()  # TODO use or not use checkpoints
model.load_state_dict(torch.load('saved_models/train_best_model.pt'))
model.eval()  # EVALING!
FG = FeatureGenerator()
summary(model, input_size=[(2, 170, FG.num_points, FG.num_axes), (2, 20)], dtypes=[torch.float32, torch.long])
model.train()  # TRAINING!

train_dataloader = get_dataloader(save_path='proc_data/test_train.npz', batch_size=batch_size, shuffle=True)
val_dataloader = get_dataloader(save_path='proc_data/test_val.npz', batch_size=batch_size, shuffle=False)

EPOCHS = 80
SWA_EPOCHS = 20
WARMDOWN_EPOCHS = 20
WARMDOWN_FACTOR = 0.1

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

train(model, optimizer, train_dataloader, epochs=EPOCHS, swa_epochs=SWA_EPOCHS, warmdown_epochs=WARMDOWN_EPOCHS,
      warmdown_factor=WARMDOWN_FACTOR, val_dataloader=val_dataloader)