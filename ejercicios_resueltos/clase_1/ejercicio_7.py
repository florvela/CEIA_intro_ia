import numpy as np
import sys
sys.path.append('..')
from utils.centroid_distances import get_centroid_distances


"""
#### Ejecicio #7:    Etiquetar Cluster
Obtener para cada fila en _X_, el índice de la fila en _C_ con distancia euclídea más pequeña. 
Es decir, para cada fila en _X_, determinar a qué cluster pertenece en C.
_Hint_: usar np.argmin.
"""

if __name__ == "__main__":

    c = [[1, 0, 0], [0, 1, 1]]  
    x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    distances = get_centroid_distances(c, x)

    clusters = np.argmin(distances, axis=0)

    print(clusters)