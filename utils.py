import os
import sys
import time
import json
import polars as pl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import GroupShuffleSplit
import editdistance

POINTS_PER_FRAME = 543


def save_arrs(arrs, path, verbose=True):
    with open(path, 'w+') as file:
        file.truncate()
    time_start = time.time()
    np.savez(path, *arrs)
    time_end = time.time()
    if verbose:
        print("Saved np arrays to {} | time spent: {:.2f} s".format(path, time_end - time_start))


def load_arrs(path, verbose=True):
    time_start = time.time()
    with np.load(path, allow_pickle=True) as arrs:
        arrs = [arrs['arr_' + str(i)] for i in range(len(arrs))]
    time_end = time.time()
    if verbose:
        print("Loaded np arrays from {} | time spent: {:.2f} s".format(path, time_end - time_start))
    return arrs


def get_meta():
    train_meta = pl.scan_csv('raw_data/train.csv')
    sup_meta = pl.scan_csv('raw_data/supplemental_metadata.csv')
    return pl.concat([train_meta, sup_meta])


def get_seq_ids():
    seq_ids = get_meta().select('sequence_id').collect().to_numpy().flatten()
    return seq_ids[seq_ids != 435344989]


def get_random_seq_ids():  # for eda/debugging
    return shuffle(get_seq_ids(), random_state=367)[:2000]


def train_val_split(seq_ids=None):
    if seq_ids is None:
        seq_ids = get_seq_ids()
    part_ids = get_part_ids(seq_ids)
    train_idxs, val_idxs = next(GroupShuffleSplit(n_splits=1, test_size=0.1,
                                                  random_state=89).split(seq_ids, seq_ids, part_ids))
    train_idxs, val_idxs = shuffle(train_idxs, random_state=376), shuffle(val_idxs, random_state=162)
    return seq_ids[train_idxs], seq_ids[val_idxs]


def get_part_ids(seq_ids):
    filtered_meta = get_meta().filter(pl.col('sequence_id').is_in(seq_ids))
    part_ids = filtered_meta.select('participant_id').collect().to_numpy().flatten()
    meta_seq_ids = filtered_meta.select('sequence_id').collect().to_numpy().flatten()
    part_ids[np.argsort(seq_ids, axis=0)] = part_ids[np.argsort(meta_seq_ids, axis=0)]
    return part_ids


def get_paths(seq_ids):
    filtered_meta = get_meta().filter(pl.col('sequence_id').is_in(seq_ids))
    paths = filtered_meta.select('path').collect().to_numpy().flatten()
    meta_seq_ids = filtered_meta.select('sequence_id').collect().to_numpy().flatten()
    paths[np.argsort(seq_ids, axis=0)] = paths[np.argsort(meta_seq_ids, axis=0)]
    full_paths = ['raw_data/' + path for path in paths.tolist()]
    return full_paths


def get_phrases(seq_ids):
    filtered_meta = get_meta().filter(pl.col('sequence_id').is_in(seq_ids))
    phrases = filtered_meta.select('phrase').collect().to_numpy().flatten()
    meta_seq_ids = filtered_meta.select('sequence_id').collect().to_numpy().flatten()
    phrases[np.argsort(seq_ids, axis=0)] = phrases[np.argsort(meta_seq_ids, axis=0)]
    return phrases


def phrases_to_labels(phrases):
    with open('raw_data/character_to_prediction_index.json') as file:
        idx_dict = json.load(file)
    labels = []
    for phrase in phrases:
        label = np.empty(len(phrase))
        for i, char in enumerate(phrase):
            label[i] = idx_dict[char]
        labels.append(label)
    return labels


def label_to_phrase(label):
    with open('raw_data/character_to_prediction_index.json') as file:
        idx_dict = json.load(file)
    char_list = [' ' for _ in range(59)]
    for char, idx in idx_dict.items():
        char_list[idx] = char
    phrase = ''
    for idx in label:
        phrase += char_list[idx]
    return phrase


def accuracy_score(outputs, labels):  # works with torch tensors TODO see if there is more efficient edit dist impl
    len_sum, dist_sum = 0, 0
    for output, label in zip(outputs, labels):
        len_sum += len(label)
        dist_sum += editdistance.eval(output.tolist(), label.tolist())
    return (len_sum - dist_sum) / len_sum
