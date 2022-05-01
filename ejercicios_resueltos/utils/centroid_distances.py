import numpy as np


def get_centroid_distances(c, x):
    x = np.array(x) 
    c = np.array(c)
    c_temp = c[:,np.newaxis]
    # get distance to X points
    substraction = c_temp - x
    sum_of_squares = np.sum((substraction) ** 2, axis=2)
    return np.sqrt(sum_of_squares)