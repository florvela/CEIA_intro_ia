import numpy as np
import sys 
sys.path.append('..')
from utils.lpnorm import LP_norm

"""
#### Ejecicio #1:    Operaciones Matriciales
Dada una matriz en formato numpy array, donde cada fila de la matriz representa un vector matemático: 
* Computar las normas l0, l1, l2, l-infinito
"""

if __name__ == "__main__":
    m = np.array([[5, 9, 6],
                  [4, 7, 9],
                  [7, 2, 1]])

    lp_norm = LP_norm()

    l0 = lp_norm.get_l0(m)
    np.testing.assert_array_equal(l0, np.array([3,3,3]))

    l1 = lp_norm.get_l1(m)
    np.testing.assert_array_equal(l1, np.array([20,20,10]))

    l2 = lp_norm.get_l2(m)
    
    linf = lp_norm.get_linf(m)
    np.testing.assert_array_equal(linf, np.array([9,9,7]))