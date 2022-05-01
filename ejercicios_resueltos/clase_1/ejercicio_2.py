import numpy as np
import sys 
sys.path.append('..')
from utils.lpnorm import LP_norm

"""
#### Ejecicio #2:    
Sorting 
Dada una matriz en formato numpy array, donde cada fila de la matriz representa un vector matemático, 
se requiere computar la norma l2 de cada vector.

Una vez obtenida la norma, se debe ordenar las mísmas de mayor a menor. 
Finalmente, obtener la matriz original ordenada por fila según la norma l2.

_Todas las operaciones debe ser vectorizadas._

"""


class Sorting(LP_norm):

    def sort_l2(self, m):
        l2 = self.get_l2(m)
        arg_sort = np.argsort(l2 * -1)
        return m[arg_sort, :]


if __name__ == "__main__":

    m = np.array([[5, 9, 6],
                  [4, 7, 9],
                  [7, 2, 1]])

    sorting_obj = Sorting()

    sorted_m_test = np.array([[4, 7, 9],
                               [5, 9, 6],
                               [7, 2, 1]])

    sorted_m = sorting_obj.sort_l2(m)
    np.testing.assert_array_equal(sorted_m, sorted_m_test)