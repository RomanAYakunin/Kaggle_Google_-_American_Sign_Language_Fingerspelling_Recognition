import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

from dataset import get_seqs, FeatureGenerator
from utils import get_seq_ids, get_random_seq_ids, get_paths, get_phrases, get_part_ids


def check_sorted():
    train_paths = [f'raw_data/train_landmarks/{file}' for file in os.listdir('raw_data/train_landmarks')]
    sup_paths = [f'raw_data/supplemental_landmarks/{file}' for file in os.listdir('raw_data/supplemental_landmarks')]
    for path in tqdm(train_paths, file=sys.stdout):
        seq_ids = pl.scan_parquet(path).select('sequence_id').collect().to_numpy().flatten()
        if (seq_ids != np.sort(seq_ids)).any():
            print('SEQ IDS NOT SORTED')
            return
    for path in tqdm(sup_paths, file=sys.stdout):
        seq_ids = pl.scan_parquet(path).select('sequence_id').collect().to_numpy().flatten()
        if (seq_ids != np.sort(seq_ids)).any():
            print('SEQ IDS NOT SORTED')
            return
    print('SEQ IDS ARE SORTED')

    print('----------------------------------------------------------'
          '-----------------------------------------------------------')


def show_num_seq():
    num_train_seqs = len(pl.scan_csv('raw_data/train.csv').select('sequence_id').collect())
    num_sup_seqs = len(pl.scan_csv('raw_data/supplemental_metadata.csv').select('sequence_id').collect())
    print('num train seqs:', num_train_seqs)
    print('num sup seqs:', num_sup_seqs)
    print('total num seqs:', num_train_seqs + num_sup_seqs)
    
    print('----------------------------------------------------------'
          '-----------------------------------------------------------')


def show_part_ids():
    part_ids = np.unique(get_part_ids(get_seq_ids()))
    print('# of unique part ids:', len(part_ids))
    print('unique part ids:', part_ids)

    print('----------------------------------------------------------'
          '-----------------------------------------------------------')


def show_random_seq_stats():
    print('stats for 2000 random sequences:')
    seq_ids = get_random_seq_ids()
    paths = get_paths(seq_ids)
    seq_lens = []
    for seq_id, path in tqdm(list(zip(seq_ids, paths)), file=sys.stdout):
        seq_len = len(pl.scan_parquet(path).select('sequence_id').filter(pl.col('sequence_id') == seq_id)\
            .collect())
        seq_lens.append(seq_len)
    seq_lens = np.array(seq_lens)
    plt.hist(seq_lens, bins=50)
    plt.title('histogram of seq_lens')
    plt.show()
    print('mean seq len:', np.mean(seq_lens))
    print('std of seq lens:', np.std(seq_lens))
    print('min seq len:', np.min(seq_lens))
    print('max seq len:', np.max(seq_lens))
    print()
    
    phrases = get_phrases(seq_ids)
    phrase_lens = np.array([len(phrase) for phrase in phrases])
    plt.hist(phrase_lens, bins=np.max(phrase_lens) - np.min(phrase_lens))
    plt.title('histogram of phrase lens')
    plt.show()
    print('mean phrase len:', np.mean(phrase_lens))
    print('std of phrase lens:', np.std(phrase_lens))
    print('min phrase len:', np.min(phrase_lens))
    print('max phrase len:', np.max(phrase_lens))
    print()

    plt.scatter(seq_lens, phrase_lens, s=0.5)
    plt.title('scatter plot of phrase lens vs seq lens')
    plt.xlabel('seq lens')
    plt.ylabel('phrase lens')
    plt.show()

    seq_to_phrase_ratios = seq_lens / phrase_lens
    plt.hist(seq_to_phrase_ratios, bins=50)
    plt.title('histogram of seq to phrase ratios')
    plt.show()
    print('mean seq to phrase ratio:', np.mean(seq_to_phrase_ratios))
    print('std of seq to phrase ratios:', np.std(seq_to_phrase_ratios))
    print('min seq to phrase ratio:', np.min(seq_to_phrase_ratios))
    print('max seq to phrase ratio:', np.max(seq_to_phrase_ratios))
    
    print('----------------------------------------------------------'
          '-----------------------------------------------------------')


def show_hand_stats():
    print('hand stats:')
    seqs = get_seqs(get_seq_ids())
    left_nan = 0
    right_nan = 0
    total_nan = np.array([0, 0, 0], dtype=np.float64)
    FG = FeatureGenerator()
    left_range = FG.norm_ranges[-2]
    right_range = FG.norm_ranges[-1]
    for seq in tqdm(seqs, file=sys.stdout):
        isnan_left = int(np.all(seq[:, left_range] == 0))
        isnan_right = int(np.all(seq[:, right_range == 0]))
        left_nan += isnan_left
        right_nan += isnan_right
        total_nan[isnan_left + isnan_right] += 1
    left_nan /= len(seqs)
    right_nan /= len(seqs)
    total_nan /= len(seqs)
    print('left nan proportion:', left_nan)
    print('right nan proportion:', right_nan)
    print('total nan array:', total_nan.tolist())

    print('----------------------------------------------------------'
          '-----------------------------------------------------------')

show_hand_stats()