import numpy as np

def load_data():
    """ Load training dataset

        Returns tuple of length 4: (X_train, y_train, X_val, y_val)
        where X_train is an N_train-x-M ndarray and y_train is an N_train-x-1 ndarray and
        where X_val is an N_val-x-M ndarray and y_val is an N_val-x-1 ndarray.
    """
    X = np.load('data/regression_train_input.npy')
    y = np.load('data/regression_train_output.npy')

    N = len(y)
    N_val = 10
    N_train = N - N_val    

    X_train = X[:N_train]
    y_train = y[:N_train]
    X_val = X[N_train:]
    y_val = y[N_train:]

    return (X_train, y_train, X_val, y_val)


def kernel_boxcar(x, z, h):
    """ Return the result of applying the boxcar kernel on the two input vectors.

        x: Mx1 numpy ndarray
        z: Mx1 numpy ndarray
        
        Returns: float value after appying kernel to x and z
    """
    res = np.linalg.norm((x-z))
    bool_d = 1 if (res <= (h/2)) else 0
    return bool_d


def kernel_rbf(x, z, h):
    """ Return the result of applying the radial basis function kernel 
        on the two input vectors, given the hyperparameter h.

        x: Mx1 numpy ndarray
        z: Mx1 numpy ndarray
        gamma: float value of hyperparameter
        
        Returns: float value after appying kernel to x and z
    """
    res_vec = (-1/h**2) * np.linalg.norm(x-z)**2
    return np.exp(res_vec)

    
def predict_kernel_regression(X, X_train, y_train, kernel_function, h=0.5):
    """ Predict the output values y for the given input design matrix X.

        X: Input matrix in NxM numpy ndarray, where we want to predict the output
            for the vector in each row of X.
        X_train: Design matrix of training input in N_train-x-M numpy ndarray
        y_train: Training output in N_train-x-1 numpy array
        kernel_function: Function that takes two arguments that are each
            Mx1 numpy ndarrays and returns a float value.
        lamb: float value of regularization hyperparameter, lambda (Note, this is a 
            different hyperparameter than the hyperparameter used in RBF kernels)

        Returns: Nx1 numpy ndarray, where the i-th entry is the predicted value 
            corresponding the i-th row vector in X
    """
    n = X.shape[0]
    n_train = X_train.shape[0]
    res = np.zeros((1, n))
    for i in range(n):
        total = 0
        X_sample = X[i].T
        W = np.zeros((1, n_train))
        for j in range(n_train):
            X_sample_train = X_train[j].T
            in1, in2 = ((X_sample), (X_sample_train))
            func_val = kernel_function(in1, in2, h)
            W[0,j] = func_val
            total += func_val
    
        if total != 0:
            W = (1/total)*W
        w_arr = np.squeeze(np.asarray(W.T))
        y_arr = np.squeeze(np.asarray(y_train))
        res[0,i] = np.dot(w_arr, y_arr)
    return res.T


def mse(y, y_hat):
    err = y - y_hat
    sqerr = err**2
    return np.mean(sqerr)

