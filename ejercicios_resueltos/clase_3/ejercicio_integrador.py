import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize 
import sys
sys.path.append("..")
from utils.metrics import MSE
from utils.models import LinearRegression
import pdb


class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    @staticmethod
    def _build_dataset(path):
        dataset = np.genfromtxt(path, delimiter=';')
        return dataset
    
    def remove_nans_and_split(self, percentage=0.8, y_col_index=None, nan_method="mean", permutate_dataset=False):
        if nan_method == "mean":
            dataset = self.nan_to_mean()
        elif nan_method == "remove_rows":
            dataset = self.remove_nan_rows()
        else:
            dataset = self.dataset

        if permutate_dataset:
            permutation = np.random.permutation(dataset)
        else:
            permutation = dataset 

        if not y_col_index:
            y_col_index = len(dataset[0]) - 1

        lim_c = int(percentage*len(dataset))
        train_set, test_set = np.split(permutation, [lim_c])

        return train_set[:,:y_col_index], train_set[:,y_col_index], test_set[:,:y_col_index], test_set[:,y_col_index]

    def nan_to_mean(self):
        return np.nan_to_num(self.dataset, nan=np.nanmean(self.dataset, axis=0))


    def remove_nan_rows(self):
        return self.dataset[~np.isnan(self.dataset).any(axis=1), :]




if __name__ == '__main__':

    PCAModel = PCA(n_components=3)#, random_state=17)
    reg = LinearRegression()
    mse_ = MSE()
    data_obj = Data("clase3v2.csv")


    ########################### reemplazando nans con los means ###########################
    train_X, train_y, test_X, test_y = data_obj.remove_nans_and_split(percentage=0.8, 
                                                                      nan_method="mean")
    
    # normalize X data
    train_X_normalized = normalize(train_X)
    test_X_normalized = normalize(test_X)

    # fit PCA model with train_X data, then transform the data for training and testing
    
    PCAModel.fit(train_X_normalized)
    train_X_pca = PCAModel.transform(train_X_normalized)
    test_X_pca = PCAModel.transform(test_X_normalized)

    # create LR model, train it with X transformed by PCA
    reg.fit(train_X_pca, train_y)
    pred_y = reg.predict(test_X_pca)
    
    means_mse = mse_(test_y, pred_y)
    print("Error con mean en los nans:",means_mse)

    ########################### borrando filas con nans ###########################
    train_X, train_y, test_X, test_y = data_obj.remove_nans_and_split(percentage=0.8, 
                                                                      nan_method="remove_rows")
    
    # normalize X data
    train_X_normalized = normalize(train_X)
    test_X_normalized = normalize(test_X)

    # fit PCA model with train_X data, then transform the data for training and testing
    
    PCAModel.fit(train_X_normalized)
    train_X_pca = PCAModel.transform(train_X_normalized)
    test_X_pca = PCAModel.transform(test_X_normalized)

    # create LR model, train it with X transformed by PCA
    reg.fit(train_X_pca, train_y)
    pred_y = reg.predict(test_X_pca)
    
    remove_rows_mse = mse_(test_y, pred_y)
    print("Error eliminando filas con nans:",remove_rows_mse)


    """
    Cuando uso el metodo de means para los nans, tengo menos MSE que cuando borro las filas. 
    En alguna que otra ocasión, sucede lo opuesto, pero es por la aleatoriedad de permutar el dataset antes de hacer el split.
    """

