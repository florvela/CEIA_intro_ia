import numpy as np
import sys 
sys.path.append('..')
from utils.indexer import Indexer

"""
#### Ejecicio #3:    Indexing 
El objetivo es construir un índice para identificadores de usuarios, es decir _id2idx_ e _idx2id_.
Para ello crear una clase, donde el índice se genere en el constructor. 
Armar métodos _get_users_id_ y _get_users_idx_.

* Identificadores de usuarios : users_id = [15, 12, 14, 10, 1, 2, 1]
* Índice de usuarios : users_id = [0, 1, 2, 3, 4, 5, 4]
```
id2idx =  [-1     4     5    -1    -1    -1     -1    -1    -1    -1     3     -1      1    -1     2     0]
          [ 0     1     2     3     4     5      6     7     8     9    10     11     12    13    14    15]

id2idx[15] -> 0 ; id2idx[12] -> 1 ; id2idx[3] -> -1
idx2id[0] -> 15 ; idx2id[4] -> 1
```
"""


if __name__ == "__main__":

    users_id = [15, 12, 14, 10, 1, 2, 1]
    users_idx = [0, 1, 2, 3, 4, 5, 4]

    indexer = Indexer(users_id)
    indexes = indexer.get_users_idx(users_id)
    ids = indexer.get_users_id(users_idx)

    ids_test = np.array([ 1,  2, 10, 12, 14, 15, 14])
    indexes_test = np.array([5, 3, 4, 2, 0, 1, 0])

    np.testing.assert_array_equal(indexes, indexes_test)
    np.testing.assert_array_equal(ids, ids_test)
