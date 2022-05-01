import pdb
import numpy as np


"""
Ejercicio #3 | Reemplazar NaNs por la media de la columna.

Dado un dataset, hacer una función que utilizando numpy reemplace los 
NaNs por la media de la columna.
"""

def nan_to_num(X):
    return np.nan_to_num(X, nan=np.nanmean(X, axis=0))



if __name__ == '__main__':
    data = np.genfromtxt('clase3v2.csv', delimiter=';')
    data_clean = nan_to_num(data)
    
    print("Dataset original:\n", data)
    print("\nDataset con la media:\n", data_clean)

    assert np.any(np.isnan(data)) == True
    assert np.any(np.isnan(data_clean)) == False