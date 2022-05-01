import numpy as np
from .centroid_distances import get_centroid_distances


class KMeans():

    def __init__(self, n_clusters, max_iter=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.labels_ = None


    def get_min_distances(self, x, centroids):
        # Se calcula la distancia entre todos los puntos en X y todos 
        # los puntos en C.
        distances = get_centroid_distances(centroids, x)
        
        # Para cada punto en X se selecciona el centroide más cercano de C.
        min_distances = np.argmin(distances, axis=0)

        return min_distances


    def k_means_loop(self, x, centroids):

        min_distances = self.get_min_distances(x, centroids)

        # Se recalculan los centroides C a partir de usar las filas de X 
        # que pertenecen a cada centroide.

        for i, centroid in enumerate(centroids):
            # print("old centroid:", centroid)
            new_centroid = np.mean(x[min_distances == i,:], axis=0)
            # print("new centroid:", new_centroid)
            centroids[i] = new_centroid
        
        return centroids, min_distances


    def fit(self, X):
        # Se seleccionan n elementos aleatorios de X como posiciones iniciales del los centroides C.
        random_x_elements = np.random.randint(0, X.shape[0], self.n_clusters)
        centroids = X[random_x_elements]

        for i in range(self.max_iter):
            centroids, min_distances = self.k_means_loop(X, centroids)

        self.cluster_centers_ = centroids
        self.labels_ = min_distances

        return self

    def predict(self, X):
        return self.get_min_distances(X, self.cluster_centers_)