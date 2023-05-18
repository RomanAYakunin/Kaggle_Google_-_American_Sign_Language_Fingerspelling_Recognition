import sys
from tqdm import tqdm
import polars as pl
import numpy as np
from utils import get_seq_ids, get_random_seq_ids, get_paths, get_phrases
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def get_num_seq():
    num_train_seqs = len(pl.scan_csv('raw_data/train.csv').select('sequence_id').collect())
    num_sup_seqs = len(pl.scan_csv('raw_data/supplemental_metadata.csv').select('sequence_id').collect())
    print('num train seqs:', num_train_seqs)
    print('num sup seqs:', num_sup_seqs)
    print('total num seqs:', num_train_seqs + num_sup_seqs)
    
    print('----------------------------------------------------------'
          '-----------------------------------------------------------')


def get_random_seq_stats():
    print('stats for 2000 random sequences:')
    seq_ids = get_random_seq_ids()
    paths = get_paths(seq_ids)
    seq_lens = []
    for seq_id, path in tqdm(list(zip(seq_ids, paths)), file=sys.stdout):
        seq_len = len(pl.scan_parquet(f'raw_data/{path}').select('sequence_id').filter(pl.col('sequence_id') == seq_id)\
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


get_random_seq_stats()
