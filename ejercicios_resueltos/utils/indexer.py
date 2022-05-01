import numpy as np


class Indexer(object):
    def __init__(self, ids):
        ids = np.unique(ids)
        id2idx = np.ones(ids.max() + 1, dtype=np.int64) * -1
        id2idx[ids] = np.arange(ids.size)
        self.id2idx = id2idx
        self.idx2id = ids

    def get_users_idx(self, ids):
        ids = self.id2idx[ids]
        return ids

    def get_users_id(self, idxs):
        return self.idx2id[idxs]