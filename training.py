import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import accuracy_score
from torch.cuda.amp import GradScaler
from augmentation import AugmentBatch


def train(model, train_dataloader, epochs, optimizer, scheduler=None,
          val_dataloader=None, eval_wait=1, save_path=None):  # TODO add augmentation
    num_batches = len(train_dataloader)
    best_val_acc = 0
    scaler = GradScaler()
    augment_batch = AugmentBatch().cuda()
    for epoch in range(1, epochs + 1):
        loss_sum = 0
        len_sum = 0
        num_samples = 0
        inf_count = 0
        for batch_i, batch in enumerate(pbar := tqdm(train_dataloader, file=sys.stdout)):
            pbar.set_description(f'epoch {epoch}/{epochs}')
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                losses, label_lengths = [], []
                for x, y, xlen, ylen in batch:
                    len_sum += len(x) * x.shape[1]
                    num_samples += len(x)
                    x, y, xlen, ylen = x.cuda(), y.cuda(), xlen.cuda(), ylen.cuda()
                    x = augment_batch(x)  # AUGMENTING !!!
                    output = model(x)
                    losses.append(F.ctc_loss(log_probs=F.log_softmax(output, dim=-1).transpose(0, 1),
                                             targets=y.to(torch.long),
                                             input_lengths=xlen.to(torch.long), target_lengths=ylen.to(torch.long),
                                             reduction='none', zero_infinity=True))
                    inf_count += torch.sum(losses[-1] == 0)
                    label_lengths.append(ylen)
                loss = torch.cat(losses)
                label_lengths = torch.cat(label_lengths)
                loss = (loss / label_lengths).mean()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # for correct mean loss tracking
            scaler.step(optimizer)
            scaler.update()
            loss = loss.detach().cpu().numpy()
            loss_sum += loss
            loss_mean = loss_sum / (batch_i + 1)
            if batch_i + 1 == num_batches and val_dataloader is not None and epoch % eval_wait == 0:
                model.eval()
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs, labels, output_lengths, label_lengths = [], [], [], []
                    for val_batch in val_dataloader:
                        for x, y, xlen, ylen in val_batch:
                            x, y, xlen, ylen = x.cuda(), y.cuda(), xlen.cuda(), ylen.cuda()
                            output = torch.argmax(model(x), dim=-1)
                            outputs.append(output)
                            labels.append(y)
                            output_lengths.append(xlen)
                            label_lengths.append(ylen)
                    label_lengths = torch.cat(label_lengths)
                    labels = torch.cat(labels).detach().split(label_lengths.detach().tolist())
                    output_lengths = torch.cat(output_lengths).detach()
                    output_list = []
                    for output_chunk in outputs:
                        for output in output_chunk:
                            output_list.append(output)
                    outputs = []
                    for output, output_length in zip(output_list, output_lengths):
                        outputs.append(output[:output_length])
                model.train()
                val_acc = accuracy_score(outputs, labels)
                pbar.set_postfix_str(f'mean train loss = {loss_mean:.4f}, val acc = {val_acc:.4f} | '
                                     f'mean sample len = {len_sum / num_samples:.4f}, '
                                     f'inf loss rate = {inf_count / num_samples:.4f}')
                if val_acc > best_val_acc and save_path is not None:
                    torch.save(model.state_dict(), save_path)
                    best_val_acc = val_acc
            else:
                pbar.set_postfix_str(f'mean train loss = {loss_mean:.4f} | '
                                     f'mean sample len = {len_sum / num_samples:.4f}, '
                                     f'inf loss rate = {inf_count / num_samples:.4f}')
        if scheduler is not None:
            scheduler.step()
    if val_dataloader is not None:
        return best_val_acc