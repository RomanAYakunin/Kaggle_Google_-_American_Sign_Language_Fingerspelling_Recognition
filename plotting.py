import torch
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from utils import get_paths, load_arrs, save_arrs
from sklearn.utils import shuffle
from dataset import FeatureGenerator, get_seqs
import json
from copy import deepcopy
from utils import get_random_seq_ids, get_phrases
from augmentation import AugmentBatch


def plot_sample(x, phrase, save_path):  # x.shape = [num_frames, num_features]
    print(x.shape)
    print(phrase)
    x = deepcopy(x)
    fig, ax = plt.subplots()
    fig.suptitle(f'{phrase}, seq len = {x.shape[0]}')
    x[:, :, 1] *= -1 * 1.5/0.9096226349071285  # TODO modify this
    plt.gcf().set_dpi(300)
    def plot_sample_frame(frame_idx):
        ax.clear()
        frame_x, frame_y = x[frame_idx, :, 0], x[frame_idx, :, 1]
        ax.set_xlim(left=-3, right=3)
        ax.set_ylim(top=2, bottom=-4)
        ax.scatter(frame_x, frame_y, s=0.5)
        for i in range(len(frame_x)):
            ax.annotate(i, (frame_x[i], frame_y[i]), fontsize=3)
    anim = FuncAnimation(fig, func=plot_sample_frame, frames=len(x))
    anim.save(save_path, writer=PillowWriter(fps=30))


seq_id = get_random_seq_ids()[1]
seq = get_seqs([seq_id])[0]
phrase = get_phrases([seq_id])[0]

plot_sample(seq, phrase, save_path='plots/sample.gif')


def augment(x):
    x = torch.from_numpy(x).unsqueeze(0).cuda()
    x = AugmentBatch().cuda()(x)
    return x.squeeze().cpu().numpy()


plot_sample(augment(seq), phrase, save_path='plots/augmented.gif')
