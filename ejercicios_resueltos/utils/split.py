import numpy as np
# import pdb

def split_X_y(X, y, percentage=0.8):
    indices = np.arange(len(X))
    permutation = np.random.permutation(indices)

    lim_c = int(percentage*len(X))
    train_set_indices, test_set_indices = np.split(permutation, [lim_c])
    # pdb.set_trace()
    
    train_X = X[train_set_indices]
    train_y = y[train_set_indices]
    
    test_X = X[test_set_indices]
    test_y = y[test_set_indices]
    
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    return train_X, train_y, test_X, test_y