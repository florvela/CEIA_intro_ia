import numpy as np

import pdb

"""
Ejercicio #1: Z-Score

Muchos algoritmos de Machine Learning necesitan datos de entrada centrados y normalizados. 
Una normalización habitual es el z-score, que implica restarle la media y dividir por el desvío a cada feature de mi dataset.

Dado un dataset X de n muestras y m features, implementar un método en numpy para normalizar con z-score. 
Pueden utilizar np.mean() y np.std().
"""

def z_score(X):
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0) 
    return (X - mean) / std


if __name__ == '__main__':
    original_data = np.genfromtxt('clase3v2.csv', delimiter=';')
    normalized_data = z_score(original_data)

    print('Original Data:')
    print(original_data)

    print('\nNormalized Data:')
    print(normalized_data)

