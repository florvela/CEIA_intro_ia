import numpy as np
import sys
sys.path.append('..')
from utils.kmeans import KMeans


"""
#### Ejercicio #8:   Implementación Básica de K-means
K-means es uno de los algoritmos más básicos en Machine Learning no supervisado.
Es un algoritmo de clusterización, que agrupa datos que comparten características similares.
Recordemos que entendemos datos como _n_ realizaciones del vector aleatorio _X_.

El algoritmo funciona de la siguiente manera:
1. El usuario selecciona la cantidad de clusters a crear _n_.
2. Se seleccionan _n_ elementos aleatorios de _X_ como posiciones iniciales del los centroides _C_.
3. Se calcula la distancia entre todos los puntos en _X_ y todos los puntos en _C_.
4. Para cada punto en _X_ se selecciona el centroide más cercano de _C_.
5. Se recalculan los centroides _C_ a partir de usar las filas de _X_ que pertenecen a cada centroide. 
6. Se itera entre 3 y 5 una cantidad fija de veces o hasta que la posición de los centroides no cambie dada una tolerancia.

Se debe por lo tanto implementar la función k_means(X, n) de manera tal que, al finalizar, devuelva la posición de los centroides
y a qué cluster pertenece cada fila de _X_. 

_Hint_: para (2) utilizar funciones de np.random, para (3) y (4) usar los ejercicios anteriores, 
para (5) es válido utilizar un for. Iterar 10 veces entre (3) y (5).  
"""


if __name__ == "__main__":

    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    X = np.array(X)

    kmeans = KMeans(n_clusters=4).fit(X)
    
    print("\nLabels:")
    print(kmeans.labels_, "\n")

    print("\nCluster Centers:")
    print(kmeans.cluster_centers_, "\n")

    arr = [4, 5, 7]
    print("\nPrediction of: ", arr)
    pred = kmeans.predict(arr)
    print(pred, "\n")

        