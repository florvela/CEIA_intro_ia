import numpy as np


"""
Ejercicio #4 | Dado un dataset X separarlo en 70 / 20 / 10

Como vimos en el ejercicio integrador, en problemas de Machine Learning es fundamental que 
separemos los datasets de n muestras, en 3 datasets de la siguiente manera:

● Training dataset: los datos que utilizaremos para entrenar nuestros modelos. 
Ej: 70% de las muestras.

● Validation dataset: los datos que usamos para calcular métricas y ajustar los hiperparámetros 
de nuestros modelos. Ej: 20% de las muestras.

● Testing dataset: una vez que entrenamos los modelos y encontramos los hiperparámetros óptimos de
los mísmos, el testing dataset se lo utiliza para computar las métricas finales de nuestros modelos
y analizar cómo se comporta respecto a la generalización. Ej: 10% de las muestras.

A partir de utilizar np.random.permutation, hacer un método que dado un dataset, 
devuelva los 3 datasets como nuevos numpy arrays.

"""


def split_train_test_val(X):
    permutation = np.random.permutation(X)

    """ 
    Si sections es [0.7*len(X), 0.9*len(X)], se divide el arr en 3 secciones:
    arr[:0.7*len(X)]                acumula el 70%
    arr[0.7*len(X):0.9*len(X)]      acumula del 70% al 90% (20%)
    arr[0.9*len(X):]                acumula del 90% al 100% (10%)
    """
    lim_a = int(0.7*len(X))
    lim_b = int(0.9*len(X))

    train_set, valid_set, test_set = np.split(permutation, [lim_a, lim_b])
    
    return train_set, valid_set, test_set 



if __name__ == '__main__':

    full_data = np.genfromtxt('clase3v2.csv', delimiter=';')
    print("Dataset original:\n", full_data)

    train_set, valid_set, test_set  = split_train_test_val(full_data) 

    print("\nTrain:\n",train_set)    
    print("\nValidation:\n",valid_set)    
    print("\nTest:\n",test_set)    

    assert full_data.shape[0] == 100
    assert train_set.shape[0] == 70
    assert valid_set.shape[0] == 20
    assert test_set.shape[0] == 10