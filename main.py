import polars as pl
import numpy as np
from utils import get_meta, get_seq_ids, get_paths, get_phrases, get_random_seq_ids, \
    slow_get_seqs, phrases_to_labels, get_part_ids, train_val_split
from dataset import NPZDataset

# seq_ids = get_seq_ids()
# seq_ids = np.array([1817370426, 1535530550, 1817123330, 1536952192, 1536750016, 1818394131, 1817882602, 1683353194])
#
# phrases = get_phrases(seq_ids)
# print(len(phrases))
# print(phrases)
#
# paths = get_paths(seq_ids)
# print(len(paths))
# print(paths)

# train_seq_ids = pl.scan_csv('raw_data/train.csv').select('sequence_id').collect().to_numpy().flatten()
# sup_seq_ids = pl.scan_csv('raw_data/supplemental_metadata.csv').select('sequence_id').collect().to_numpy().flatten()
# print(len(train_seq_ids))
# print(train_seq_ids[:5])
# print(len(sup_seq_ids))
# print(sup_seq_ids[:5])
# print("-------")
# print(np.intersect1d([3, 1, 2, 5, 6], [6, 10, 237, 2, 5, 173]))
# print(np.intersect1d(train_seq_ids, sup_seq_ids))

# seq_ids = get_random_seq_ids()[:200]
# # seq_ids = [1816796431]
# # seqs = get_seqs(seq_ids)
# slow_seqs = slow_get_seqs(seq_ids)
# print(len(slow_seqs))
# seqs = get_seqs(seq_ids)
# print(len(seqs))
# are_equal = True
# for i in range(len(seqs)):
#     if not ((slow_seqs[i] == seqs[i]) | np.isnan(slow_seqs[i]) | np.isnan(seqs[i])).all():
#         print('NOT EQUAL!')
#         are_equal = False
#         break
# if are_equal:
#     print('ARE EQUAL!')


# columns = pl.scan_csv('plots/csv_examples/full.csv').columns
# print(columns[1:-1])

# seq_ids = [1817370426, 1535467051, 1816796431]
# # print(phrases_to_labels(get_phrases(seq_ids)))
# print(get_part_ids(seq_ids))

# train_seq_ids, val_seq_ids = train_val_split()
# print(len(train_seq_ids))
# print(len(np.unique(get_part_ids(train_seq_ids))))
# print(len(val_seq_ids))
# print(len(np.unique(get_part_ids(val_seq_ids))))
# print(np.intersect1d(get_part_ids(train_seq_ids), get_part_ids(val_seq_ids)))
# print(np.intersect1d([10, 1, 2, 3], [5, 2, 7, 1]))

# path = get_paths([1816796431])[0]
# pl.scan_parquet(path).head(10).collect().write_csv('plots/full.csv')

seq_ids = get_random_seq_ids()[:5]
print(seq_ids)
# NPZDataset.create(seq_ids=seq_ids, save_path='proc_data/test.npz')
dataset = NPZDataset('proc_data/test.npz')
print(len(dataset))
print(dataset[3])