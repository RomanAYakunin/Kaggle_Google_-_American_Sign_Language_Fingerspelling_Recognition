import os
import sys
import time
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils import shuffle


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
    return get_meta().select('sequence_id').collect().to_numpy().flatten()


def get_random_seq_ids():  # for eda/debugging
    return shuffle(get_seq_ids(), random_state=367)[:2000]


def get_paths(seq_ids):
    meta = get_meta()
    filtered_meta = meta.filter(pl.col('sequence_id').is_in(seq_ids))
    paths = filtered_meta.select('path').collect().to_numpy().flatten()
    meta_seq_ids = filtered_meta.select('sequence_id').collect().to_numpy().flatten()
    paths[np.argsort(seq_ids, axis=0)] = paths[np.argsort(meta_seq_ids, axis=0)]
    return paths


def get_phrases(seq_ids):
    meta = get_meta()
    filtered_meta = meta.filter(pl.col('sequence_id').is_in(seq_ids))
    phrases = filtered_meta.select('phrase').collect().to_numpy().flatten()
    meta_seq_ids = filtered_meta.select('sequence_id').collect().to_numpy().flatten()
    phrases[np.argsort(seq_ids, axis=0)] = phrases[np.argsort(meta_seq_ids, axis=0)]
    return phrases


# def get_signs(paths):  # more efficient than getting signs one by one from polars LazyFrame
#     cropped_paths = ['/'.join(path.split('/')[1:]) for path in paths]
#     signs = pl.scan_csv('raw_data/train.csv').filter(pl.col('path').is_in(cropped_paths)).select('sign') \
#         .collect().to_numpy().flatten()
#     train_df_paths = pl.scan_csv('raw_data/train.csv').select('path').filter(pl.col('path').is_in(cropped_paths)) \
#         .collect().to_numpy().flatten()  # is in different order from cropped_paths
#     signs[np.argsort(cropped_paths)] = signs[np.argsort(train_df_paths)]  # matches via sort, then unsorts
#     return signs