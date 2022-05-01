import numpy as np


class PCA():
    
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, X):
        standardized_X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) 
        
        cov = np.cov(standardized_X.T)
        
        eigen_values, eigen_vectors = np.linalg.eig(cov)
        print("Eigenvector: \n",eigen_vectors,"\n")
        print("Eigenvalues: \n", eigen_values, "\n")
        
        eigen_values_indexes = np.argsort(eigen_values*-1)
        eigen_values = eigen_values[eigen_values_indexes]
        eigen_vectors = eigen_vectors[:,eigen_values_indexes]
        
        return standardized_X.dot(eigen_vectors[:,:self.n_components])