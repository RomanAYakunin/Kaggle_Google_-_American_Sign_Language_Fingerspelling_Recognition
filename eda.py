import os
import sys

from tqdm import tqdm
import polars as pl
import numpy as np
from utils import get_seq_ids, get_random_seq_ids, get_paths, get_phrases, get_part_ids
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from dataset import get_seqs, FeatureGenerator


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


def show_seq_stats(mode='all'):
    print(f'stats for {mode} sequences:\n')
    seq_ids = get_seq_ids(mode=mode)
    seqs = get_seqs(seq_ids)
    seq_lens = np.array([len(seq) for seq in seqs])
    plt.hist(seq_lens, bins=np.max(seq_lens) - np.min(seq_lens))
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
    plt.hist(seq_to_phrase_ratios, bins=200)
    plt.title('histogram of seq to phrase ratios')
    plt.show()
    print('mean seq to phrase ratio:', np.mean(seq_to_phrase_ratios))
    print('std of seq to phrase ratios:', np.std(seq_to_phrase_ratios))
    print('min seq to phrase ratio:', np.min(seq_to_phrase_ratios))
    print('max seq to phrase ratio:', np.max(seq_to_phrase_ratios))
    
    print('----------------------------------------------------------'
          '-----------------------------------------------------------')


def show_hand_stats(mode='all'):
    print(f'hand stats for {mode} sequences:\n')
    seq_ids = get_seq_ids(mode=mode)
    seqs = get_seqs(seq_ids)
    left_nan = 0
    right_nan = 0
    total_nan = np.array([0, 0, 0], dtype=np.float64)
    both_hands_count = 0
    one_dom_ratios = []
    FG = FeatureGenerator()
    left_range = FG.norm_ranges[-2]
    right_range = FG.norm_ranges[-1]
    for i, seq in enumerate(tqdm(seqs, file=sys.stdout)):
        isnan_left = int(np.all(seq[:, left_range[0]: left_range[1]] == 0))
        isnan_right = int(np.all(seq[:, right_range[0]: right_range[1]] == 0))
        left_nan += isnan_left
        right_nan += isnan_right
        total_nan[isnan_left + isnan_right] += 1
        if isnan_left + isnan_right == 0:
            both_hands_count += 1
            left_not_nan = np.sum(np.any(seq[:, left_range[0]: left_range[1]] != 0, axis=(1, 2)).astype(np.float64))
            right_not_nan = np.sum(np.any(seq[:, right_range[0]: right_range[1]] != 0, axis=(1, 2)).astype(np.float64))
            neither_nan = left_not_nan + right_not_nan
            one_dom_ratios.append(max(left_not_nan/neither_nan, right_not_nan/neither_nan))
    left_nan /= len(seqs)
    right_nan /= len(seqs)
    total_nan /= len(seqs)
    print('left nan proportion:', left_nan)
    print('right nan proportion:', right_nan)
    print('total nan array:', total_nan.tolist())
    if both_hands_count != 0:
        one_dom_ratios = np.array(one_dom_ratios)  # TODO get examples for various ratios
        print('one dom ratios mean:', np.mean(one_dom_ratios))
        print('one dom ratios std:', np.std(one_dom_ratios))
        print('one dom ratios min:', np.min(one_dom_ratios))
        print('one dom ratios max:', np.max(one_dom_ratios))

        plt.hist(one_dom_ratios, bins=200)
        plt.title('histogram of one dom ratios')
        plt.show()

        counts, bins = np.histogram(one_dom_ratios, bins=200)
        counts = np.cumsum(counts)
        plt.stairs(counts, bins, fill=True)
        plt.title('cumulative histogram of one dom ratios')
        plt.show()

    print('----------------------------------------------------------'
          '-----------------------------------------------------------')


mode = 'all'
# show_seq_stats(mode)
show_hand_stats(mode)
