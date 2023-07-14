import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from augmentation import AugmentX, AugmentY
from utils import label_to_phrase
import editdistance


def get_accuracy_scores(outputs, labels):  # works with torch tensors
    len_sums, dist_sums = [0, 0], [0, 0]
    for output, label in zip(outputs, labels):
        type = label[0] == 61  # train -> 0, supp -> 1
        len_sums[type] += len(label)
        dist_sums[type] += editdistance.eval(output.tolist(), label.tolist())
    train_acc = (len_sums[0] - dist_sums[0]) / len_sums[0]
    supp_acc = (len_sums[1] - dist_sums[1]) / len_sums[1]
    acc = (sum(len_sums) - sum(dist_sums)) / sum(len_sums)
    return train_acc, supp_acc, acc


def train(model, train_dataloader, epochs, optimizer, scheduler=None,  # TODO add scheduler
          val_dataloader=None, eval_wait=1, save_path=None):
    num_batches = len(train_dataloader)
    best_train_acc = 0  # train acc = accuracy on val samples that comes from the train folder
    scaler = GradScaler()
    augment_x, augment_y = AugmentX().cuda(), AugmentY(train_dataloader).cuda()
    output_examples = []
    for epoch in range(1, epochs + 1):
        loss_sum = 0
        len_sum = 0
        num_samples = 0
        for batch_i, batch in enumerate(pbar := tqdm(train_dataloader, file=sys.stdout)):
            pbar.set_description(f'epoch {epoch}/{epochs}')
            with torch.autocast(device_type='cuda', dtype=torch.float16):  # Check if dtype is needed TODO NOW!
                losses = []
                for x, y in batch:
                    len_sum += len(x) * x.shape[1]
                    num_samples += len(x)
                    x, y, = x.cuda(), y.cuda()
                    x, noise_y = augment_x(x), augment_y(y)  # AUGMENTING !!!  # TODO try removing padding token
                    loss = F.cross_entropy(input=model(x, noise_y)[:, :-1].transpose(1, 2), target=y[:, 1:],
                                           label_smoothing=0.2, ignore_index=62, reduction='none')
                    losses.append(loss.sum(dim=1))  # TODO try not normalizing by len
                loss = torch.cat(losses).mean()
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
                            model_outputs = model.infer(x, sot=y[:, 0])
                            for output in model_outputs:
                                outputs.append(output[:torch.argwhere(output == 59).ravel()[0]])
                            for label in y:
                                labels.append(label[:torch.argwhere(label == 59).ravel()[0]])
                model.train()
                train_acc, supp_acc, acc = get_accuracy_scores(outputs, labels)
                pbar.set_postfix_str(f'mean train loss = {loss_mean:.4f} | '
                                     f'train acc = {train_acc:.4f}, '
                                     f'supp acc = {supp_acc:.4f}, '
                                     f'acc = {acc:.4f} | '
                                     f'mean sample len = {len_sum / num_samples:.4f}')
                output_examples = [label_to_phrase(labels[0][1:].cpu().numpy()),
                                   label_to_phrase(outputs[0].cpu().numpy())]
                if train_acc > best_train_acc and save_path is not None:
                    torch.save(model.state_dict(), save_path)
                    best_train_acc = train_acc
            else:
                pbar.set_postfix_str(f'mean train loss = {loss_mean:.4f} | '
                                     f'mean sample len = {len_sum / num_samples:.4f}')
        if len(output_examples) != 0:
            print(f'label: {output_examples[0]}, output: {output_examples[1]}')
        if val_dataloader is None:
            torch.save(model.state_dict(), save_path)
        if scheduler is not None:
            scheduler.step()
    if val_dataloader is not None:
        return best_train_acc