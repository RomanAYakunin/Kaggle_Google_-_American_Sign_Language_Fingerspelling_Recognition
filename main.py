import polars as pl
import numpy as np
from utils import get_meta, get_seq_ids, get_paths, get_phrases

# seq_ids = get_seq_ids()
seq_ids = np.array([1817370426, 1535530550, 1817123330, 1536952192, 1536750016, 1818394131, 1817882602, 1683353194])

phrases = get_phrases(seq_ids)
print(len(phrases))
print(phrases)

paths = get_paths(seq_ids)
print(len(paths))
print(paths)

# train_seq_ids = pl.scan_csv('raw_data/train.csv').select('sequence_id').collect().to_numpy().flatten()
# sup_seq_ids = pl.scan_csv('raw_data/supplemental_metadata.csv').select('sequence_id').collect().to_numpy().flatten()
# print(len(train_seq_ids))
# print(train_seq_ids[:5])
# print(len(sup_seq_ids))
# print(sup_seq_ids[:5])
# print("-------")
# print(np.intersect1d([3, 1, 2, 5, 6], [6, 10, 237, 2, 5, 173]))
# print(np.intersect1d(train_seq_ids, sup_seq_ids))