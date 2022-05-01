import numpy as np


"""

#### Ejecicio #5:    Average Query Precision :house:
En information retrieval o search engines, en general contamos con queries “q” 
y para cada “q” una lista de documentos que son verdaderamente relevantes. 

Para evaluar un search engine, es común utilizar la métrica **average query precision**.

Tomando de referencia el siguiente ejemplo, calcular la métrica con NumPy utilizando operaciones vectorizadas.
```
q_id =             [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
predicted_rank =   [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3]
truth_relevance =  [T, F, T, F, T, T, T, F, F, F, F, F, T, F, F, T] 
```
* Precision para q_id 1 = 2 / 4
* Precision para q_id 2 = 3 / 3
* Precision para q_id 3 = 0 / 5
* Precision para q_id 4 = 2 / 4

**_average query precision_** = ((2/4) + (3/3) + (0/5) + (2/4)) / 4

"""


def avg_q_precision(query_ids, truth_relevance):

    true_relevance_mask = (truth_relevance == 1) # array([ True, False,  True, False,  True,  True,  True, False, False, False, False, False,  True, False, False,  True])

    filtered_query_id = query_ids[true_relevance_mask] # array([1, 1, 2, 2, 2, 4, 4])
    
    filtered_true_relevance_count = np.bincount(filtered_query_id) # array([0, 2, 3, 0, 2])

    # contar queries con 0 en queries sin documentos relevantes
    unique_query_ids = np.unique(query_ids)                             # [1 2 3 4]
    non_zero_count_idxs = np.where(filtered_true_relevance_count > 0)   # (array([1, 2, 4]),)
    true_relevance_count = np.zeros(unique_query_ids.max() + 1)         # [0. 0. 0. 0. 0.]
    
    true_relevance_count[non_zero_count_idxs] = filtered_true_relevance_count[non_zero_count_idxs] # array([0., 2., 3., 0., 2.])

    true_relevance_count_by_query = true_relevance_count[unique_query_ids] # array([2., 3., 0., 2.])

    fetched_documents_count = np.bincount(query_ids)[unique_query_ids] # array([4, 3, 5, 4])

    precision_by_query = true_relevance_count_by_query / fetched_documents_count # array([0.5, 1. , 0. , 0.5])

    return np.mean(precision_by_query) # 0.5


if __name__ == "__main__":

    T = 1
    F = 0

    truth_relevance =  [T, F, T, F, T, T, T, F, F, F, F, F, T, F, F, T] 
    truth_relevance = np.array(truth_relevance)

    q_ids = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
    query_ids = np.array(q_ids)

    answer = avg_q_precision(query_ids, truth_relevance)

    assert answer == 0.5