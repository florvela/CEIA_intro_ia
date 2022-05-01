import numpy as np
import sys 
sys.path.append('..')
from utils.metrics import Precision, Recall, Accuracy

"""
#### Ejecicio #4:    Precision, Recall, Accuracy :house:
En los problemas de clasificación, se cuenta con dos arreglos, la **verdad** (ground truth) 
y la **predicción** (prediction). 

Cada elemento de los arreglos puede tomar dos valores: _True_ (representado por 1) 
y _False_ (representado por 0). 

Por lo tanto, se pueden definir cuatro variables:
* True Positive (TP): la verdad es 1 y la predicción es 1.
* True Negative (TN): la verdad es 0 y la predicción es 0.
* False Negative (FN): la verdad es 1 y la predicción es 0.
* False Positive (FP): la verdad es 0 y la predicción es 1.

A partir de esas cuatro variables, se definen las siguientes métricas:
* Precision = TP / (TP + FP)
* Recall = TP / (TP + FN)
* Accuracy = (TP + TN) / (TP + TN + FP + FN)

Para los siguientes arreglos, representando la **verdad** y la **predicción**,
calcular las métricas anteriores con operaciones vectorizadas en NumPy.
* truth = [1,1,0,1,1,1,0,0,0,1]
* prediction = [1,1,1,1,0,0,1,1,0,0]
"""

if __name__ == "__main__":

    truth = np.array([1,1,0,1,1,1,0,0,0,1])
    predictions = np.array([1,1,1,1,0,0,1,1,0,0])

    for metric in [Precision, Recall, Accuracy]:
        aux = metric(truth=truth, predictions=predictions)
        res = aux()
        assert res == 0.5