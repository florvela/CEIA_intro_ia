import numpy as np
import sys
sys.path.append('..')
from utils.metrics import Precision, Recall, Accuracy, AvgQPrecision


"""
#### Ejercicio #9:   Computar Métricas con __call__ :house:
En problemas de machine learning, es muy común que para cada predicción que obtenemos en nuestro dataset 
de verificacion y evaluacion, almacenemos en arreglos de numpy el resultado de dicha predicción, 
junto con el valor verdadero y parámetros auxiliares (como el ranking de la predicción y el query id). 

Luego de obtener todas las predicciones, podemos utilizar la información almacenada en los arreglos de numpy, 
para calcular todas las métricas que queremos medir en nuestro sistema. 

Una buena práctica para implementar esto en Python, es crear clases que hereden de una clase Metric “base” y 
que cada métrica implemente el método __call__.

Utilizar herencia, operador __call__ y _kwargs_, para escribir un programa que permita calcular todas las 
métricas de los ejercicios anteriores mediante un for.
"""

def get_metrics(**kwargs):

    metrics = {}

    for metric in [Precision, Recall, Accuracy, AvgQPrecision]:
        aux = metric(**kwargs)
        res =  aux()
        metrics[metric.__name__] = res

    return metrics



if __name__ == "__main__":

    truth = np.array([1,1,0,1,1,1,0,0,0,1])
    predictions = np.array([1,1,1,1,0,0,1,1,0,0])
    truth_relevance = np.array([1,0,1,0,1,1,1,0,0,0,0,0,1,0,0,1])
    query_ids = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4])

    metrics = get_metrics(truth=truth, predictions=predictions, query_ids=query_ids, truth_relevance=truth_relevance)

    print(metrics)
