import numpy as np
import sys
sys.path.append('..')
from utils.centroid_distances import get_centroid_distances

"""
#### Ejecicio #6:    Distancia a Centroides
Dada una nube de puntos _X_ y centroides _C_, obtener la distancia entre
cada vector _X_ y los centroides utilizando operaciones vectorizadas y broadcasting en NumPy.
Utilizar como referencia los siguientes valores:
```
X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
C = [[1, 0, 0], [0, 1, 1]]   
```
"""

if __name__ == "__main__":
    
    c = [[1, 0, 0], [0, 1, 1]]  
    x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    distances = get_centroid_distances(c, x)

    print(distances)

    #   [[ 3.60555128  8.36660027 13.45362405]
    #   [ 2.44948974  7.54983444 12.72792206]]