import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import accuracy_score
from torch.cuda.amp import GradScaler
from augmentation import AugmentBatch
from utils import label_to_phrase


def train(model, train_dataloader, epochs, optimizer, label_smooth=0.2, scheduler=None,
          val_dataloader=None, eval_wait=1, save_path=None):  # TODO add augmentation
    num_batches = len(train_dataloader)
    best_val_acc = 0
    scaler = GradScaler()
    augment_batch = AugmentBatch().cuda()
    print_mssg = []
    for epoch in range(1, epochs + 1):
        loss_sum = 0
        len_sum = 0
        num_samples = 0
        for batch_i, batch in enumerate(pbar := tqdm(train_dataloader, file=sys.stdout)):
            pbar.set_description(f'epoch {epoch}/{epochs}')
            with torch.autocast(device_type='cuda', dtype=torch.float16):  # Check if dtype is needed TODO NOW!
                loss = 0
                batch_size = 0
                for x, y in batch:
                    len_sum += len(x) * x.shape[1]
                    num_samples += len(x)
                    batch_size += len(x)
                    x, y, = x.cuda(), y.cuda()
                    x = augment_batch(x)  # AUGMENTING !!!  # TODO add label smooth
                    chunk_loss = F.cross_entropy(input=model(x, y)[:, :-1].transpose(1, 2), target=y[:, 1:],
                                                 ignore_index=61)  # TODO try removing padding token
                    loss += chunk_loss * len(x)  # TODO consider weighing different-length sequences equally in the loss
                loss /= batch_size
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # for correct mean loss tracking
            scaler.step(optimizer)
            scaler.update()
            loss = loss.detach().cpu().numpy()
            loss_sum += loss
            loss_mean = loss_sum / (batch_i + 1)
            if batch_i + 1 == num_batches and val_dataloader is not None and epoch % eval_wait == 0:
                outputs, labels = [], []
                model.eval()
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    for val_batch in val_dataloader:
                        for x, y in val_batch:
                            x, y = x.cuda(), y.cuda()
                            model_outputs = model.infer(x)
                            for output in model_outputs:  # TODO move this part to utils so it can be reused
                                outputs.append(output[:torch.argwhere(output == 59).ravel()[0]])
                            for label in y:
                                labels.append(label[1:torch.argwhere(label == 59).ravel()[0]])
                model.train()
                print_mssg = [label_to_phrase(labels[0].cpu().numpy()), label_to_phrase(outputs[0].cpu().numpy())]
                val_acc = accuracy_score(outputs, labels)
                pbar.set_postfix_str(f'mean train loss = {loss_mean:.4f}, val acc = {val_acc:.4f} | '
                                     f'mean sample len = {len_sum / num_samples:.4f}')
                if val_acc > best_val_acc and save_path is not None:
                    torch.save(model.state_dict(), save_path)
                    best_val_acc = val_acc
            else:
                pbar.set_postfix_str(f'mean train loss = {loss_mean:.4f} | '
                                     f'mean sample len = {len_sum / num_samples:.4f}')
        print(print_mssg)  # TODO undo
        if val_dataloader is None:
            torch.save(model.state_dict(), save_path)
        if scheduler is not None:
            scheduler.step()
    if val_dataloader is not None:
        return best_val_acc