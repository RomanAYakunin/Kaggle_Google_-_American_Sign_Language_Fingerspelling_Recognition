import sys

import editdistance
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from augmentation import AugmentX, AugmentY
from utils import label_to_phrase


def get_accuracy_scores(outputs, labels):  # works with torch tensors
    len_sums, dist_sums = [0, 0], [0, 0]
    for output, label in zip(outputs, labels):
        type = label[0] == 61  # train -> 0, supp -> 1
        len_sums[type] += len(label[1:])
        dist_sums[type] += editdistance.eval(output.tolist(), label[1:].tolist())
    train_acc = (len_sums[0] - dist_sums[0]) / len_sums[0]
    supp_acc = (len_sums[1] - dist_sums[1]) / len_sums[1]
    acc = (sum(len_sums) - sum(dist_sums)) / sum(len_sums)
    return train_acc, supp_acc, acc


def train(model, optimizer, train_dataloader, epochs, swa_epochs, warmdown_epochs, warmdown_factor,
          val_dataloader=None, eval_wait=1):
    num_batches = len(train_dataloader)
    best_acc = 0  # train acc = accuracy on val samples that comes from the train folder
    scaler = GradScaler()
    augment_x, augment_y = AugmentX().cuda(), AugmentY(train_dataloader).cuda()
    warmdown_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                                gamma=warmdown_factor ** (1 / warmdown_epochs),
                                                                verbose=True)
    swa_model = None
    output_examples = []
    for curr_epoch in range(1, epochs + 1):
        if curr_epoch == epochs - warmdown_epochs - swa_epochs + 1:
            print(f'starting warmdown (epoch {curr_epoch})')
        if epochs - swa_epochs >= curr_epoch > epochs - warmdown_epochs - swa_epochs:
            warmdown_scheduler.step()
        if curr_epoch == epochs - swa_epochs + 1:
            print(f'warmdown complete')
            print(f'starting stochastic weight averaging (epoch {curr_epoch})')
        loss_sum, reg_loss_sum = 0, 0
        len_sum = 0
        aug_len_sum = 0
        num_samples = 0
        for batch_i, batch in enumerate(pbar := tqdm(train_dataloader, file=sys.stdout)):
            pbar.set_description(f'epoch {curr_epoch}/{epochs}')
            with torch.autocast(device_type='cuda'):  # Check if dtype is needed TODO NOW!
                losses = []
                for x, y in batch:
                    len_sum += x.shape[0] * x.shape[1]
                    num_samples += x.shape[0]
                    x, y, = x.cuda(), y.cuda()
                    x, noise_y = augment_x(x), augment_y(y)  # AUGMENTING !!!  # TODO try removing padding token
                    aug_len_sum += x.shape[0] * x.shape[1]
                    outputs = model(x, noise_y)
                    loss = F.cross_entropy(input=outputs[:, :-1].transpose(1, 2), target=y[:, 1:],
                                           label_smoothing=0.2, ignore_index=62, reduction='none')
                    loss = loss.sum(dim=1) / (loss != 0).sum(dim=1)  # TODO try not normalizing by len
                    losses.append(loss)
            loss = torch.cat(losses).mean()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # for correct mean loss tracking
            scaler.step(optimizer)
            scaler.update()
            # tracking loss ===================
            loss = loss.detach().cpu().numpy()
            loss_sum += loss
            loss_mean = loss_sum / (batch_i + 1)
            # ====================================
            if batch_i + 1 == num_batches and val_dataloader is not None and curr_epoch % eval_wait == 0:
                outputs, labels = [], []
                model.eval()
                with torch.inference_mode(), torch.autocast(device_type='cuda'):
                    for val_batch in val_dataloader:  # TODO TRY WITHOUT AUTOCAST!
                        for x, y in val_batch:
                            x, y = x.cuda(), y.cuda()
                            model_outputs = model.infer(x, sot=y[:, 0])
                            for output in model_outputs:
                                outputs.append(output[:torch.argwhere(output == 59).ravel()[0]])
                            for label in y:
                                labels.append(label[:torch.argwhere(label == 59).ravel()[0]])
                model.train()
                train_acc, supp_acc, acc = get_accuracy_scores(outputs, labels)
                pbar.set_postfix_str(f'loss = {loss_mean:.4f} | '
                                     f'acc = {acc:.4f}, '
                                     f'train = {train_acc:.4f}, '
                                     f'supp = {supp_acc:.4f} | '
                                     f'MSL = {round(len_sum / num_samples)}, '
                                     f'aug = {round(aug_len_sum / num_samples)}')
                output_examples = [label_to_phrase(labels[0][1:].cpu().numpy()),
                                   label_to_phrase(outputs[0].cpu().numpy())]
                if train_acc > best_acc:
                    torch.save(model.state_dict(), 'saved_models/train_best_model.pt')
                    best_acc = train_acc
            else:
                pbar.set_postfix_str(f'loss = {loss_mean:.4f} | '
                                     f'MSL = {round(len_sum / num_samples)}, '
                                     f'aug = {round(aug_len_sum / num_samples)}')
        if len(output_examples) != 0:
            print(f'label: {output_examples[0]}, output: {output_examples[1]}')
        torch.save(model.state_dict(), 'saved_models/train_last_model.pt')
        if curr_epoch > epochs - swa_epochs:
            if swa_model is None:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
            else:
                swa_model.update_parameters(model)
            torch.save(swa_model.module.state_dict(), 'saved_models/train_swa_model.pt')