import numpy as np
import pdb

"""
Ejercicio #2 | Remover filas y columnas con NaNs en un dataset
Dado un dataset, hacer una función que, utilizando numpy, filtre las columnas y las filas que tienen NaNs.
"""

def remove_nan_rows(X):
    return X[~np.isnan(X).any(axis=1), :]

def remove_nan_cols(X):
    return X[:,~(np.isnan(X).any(axis=0))]


if __name__ == '__main__':
    data = np.genfromtxt('clase3v2.csv', delimiter=';')
    data_no_nan_cols = remove_nan_cols(data)
    data_no_nan_rows = remove_nan_rows(data)

    print("Dataset original:\n", data)
    print("\nDataset sin columnas con nans:\n", data_no_nan_cols)
    print("\nDataset sin filas con nans:\n", data_no_nan_rows)

    assert np.any(np.isnan(data)) == True
    assert np.any(np.isnan(data_no_nan_cols)) == False
    assert np.any(np.isnan(data_no_nan_rows)) == False
