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
from augmentation import AugmentX


def augment(x):
    x = torch.from_numpy(x).unsqueeze(0).cuda().to(torch.float32)
    x = AugmentX().cuda()(x)
    return x.squeeze(0).cpu().numpy()


def plot_sample(x, phrase, save_path, center=False):  # x.shape = [num_frames, num_features]
    print(x.shape)
    print(phrase)
    x = deepcopy(x)
    if center:
        is_nan = (x == 0).astype(np.float32)
        x -= np.sum((1 - is_nan) * x, axis=(0, 1), keepdims=True) / \
             (np.sum((1 - is_nan), axis=(0, 1), keepdims=True) + 1e-5)
        x *= (1 - is_nan)
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


# seq_id = get_random_seq_ids()[343]
# seq = get_seqs([seq_id])[0]
# phrase = get_phrases([seq_id])[0]
#
# plot_sample(seq, phrase, save_path='plots/sample.gif', center=True)
#
# plot_sample(augment(seq), phrase, save_path='plots/augmented.gif')

from utils import get_gislr_paths, get_signs
from dataset import get_gislr_data

path = shuffle(get_gislr_paths(), random_state=413)[2]
seq = get_gislr_data([path])[0]
phrase = get_signs([path])[0]

plot_sample(seq, phrase, save_path='plots/gislr_sample.gif', center=True)

plot_sample(augment(seq), phrase, save_path='plots/gislr_augmented.gif')
