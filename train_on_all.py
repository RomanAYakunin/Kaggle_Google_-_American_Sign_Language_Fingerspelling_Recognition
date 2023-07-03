import torch
from torch_model.model import Model
from training import train
from torchinfo import summary
from dataset import FeatureGenerator, get_dataloader

# train_seq_ids = get_seq_ids()
# NPZDataset.create(train_seq_ids, 'proc_data/all.npz', crop_labels=True)

batch_size = 64
model = Model().cuda()
FG = FeatureGenerator()
summary(model, input_size=(2, FG.max_len, FG.num_points, FG.num_axes))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)  # TODO restart with proper lr

train_dataloader = get_dataloader(save_path='proc_data/all.npz', batch_size=batch_size, shuffle=True)

train(model, train_dataloader, epochs=1000, optimizer=optimizer, save_path='saved_models/test_model.pt')