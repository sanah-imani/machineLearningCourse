import numpy as np 
"""
import sys
import os
import util
"""




def load_data():
    """ Load unlabeled MNIST dataset of just zero and one digits.

        Returns an N-by-784 numpy ndarray X
    """
    N = 1000
    X = np.load('data/mnist_zeros_ones.npy').astype(np.float32)

    X = np.reshape(X[:N][:][:], (N, 784))

    X = X / 255.0

    return X

def consistent_scale_eigenvectors(V):
    """ Scale the columns of V such that everyone uses a consistent
        set of eigenvectors.

        Input:
        V: numpy ndarray each **column** is an eigenvector

        Returns V. V is modified in place and also returned.
        
        Implementation based on code from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py::svd_flip
    """
    max_abs_cols = np.argmax(np.abs(V), axis=0)
    signs = np.sign(V[max_abs_cols, range(V.shape[1])])
    V *= signs

    return V

def pca(X, K):
    """ Return the PCA projection of X onto K-dimensions
        X: numpy ndarray of shape (N, M) where each row represents a data point
        K: integer representing the desired number of output dimensions

        Don't forget to center your data first
        Use eigenvectors based on np.linalg.eig or np.linalg.svd.
        We suggested using np.linalg.svd because it returns sorted arrays.

        NOTE: In order to make the autograder happy, you must pass your eigenvectors
        into the consisten_scale_eigenvectors function above before applying them 
        to your data!

        Returns a numpy ndarray of shape (N, K)
    """
    
   
    
    N = X.shape[0]
    
    C = (X-(X.mean(axis=0)))
    # svd step
    U, S, V = np.linalg.svd(C)
    
    vectors = consistent_scale_eigenvectors(V.T)
 
    vectors = vectors[:, :K]
    #project data
    P = np.matmul(C, vectors)
    
    return P
    
def gaussian_pdf(x, mu, Sigma):
    """ Return the Gaussian pdf evaluated at x.

        x: numpy ndarray of size (M, )
        mu: numpy ndarray of size (M, )
        Sigma: numpy ndarray of size (M, M)

        returns float value of pdf at x

    """
    K = len(x)
    x_c = x - mu
    
    dterm = np.linalg.det(Sigma)
    
    res = (1/np.sqrt((2*np.pi)**K * dterm))

    return res * np.exp(-x_c @ np.linalg.inv(Sigma) @ x_c / 2)


def log_likelihood(X, pi_list, mu_list, Sigma_list):
    """ Return the log-likelihood of a GMM model given X. 

        X: (N,M) numpy ndarray
        pi_list: List of K scalar values: the marginal probablity of each cluster
        mu_list: List of K (M, ) numpy arrays, mean of each cluster
        Sigma_list: List of K (M, M) numpy arrays, covariance matrix of each cluster
        
        Returns: scalar, the log-likelihood of X
    """
    N = X.shape[0]

    ell = 0
    for i in range(len(X)):
        mixture_prob = 0
        for pi, mu, Sigma in zip(pi_list, mu_list, Sigma_list):
            mixture_prob += pi * gaussian_pdf(X[i], mu, Sigma)
        ell += np.log(mixture_prob)

    ell = ell / N

    return ell 

def update_Z(X, Z, pi_list, mu_list, Sigma_list):
    """ Update p(z_k^i = 1 | x) given the GMM parameters.
        Update Z inplace.

        X: (N,M) numpy ndarray representing training data
        Z: (N,K) numpy ndarray where Z[i, k] is the probability of the i-th point
            belonging to the k-th cluster, p(z_k^i = 1 | x, parameters) 
        pi_list: List of K scalar values: the marginal probablity of each cluster
        mu_list: List of K (M, ) numpy arrays, mean of each cluster
        Sigma_list: List of K (M, M) numpy arrays, covariance matrix of each cluster
    """

    N, K = Z.shape
    for i in range(N):
        mixture_prob = 0
        temp = np.zeros((1, K))
        for k in range(K):
            p = pi_list[k]*gaussian_pdf(X[i], mu_list[k], Sigma_list[k])
            temp[0,k] = p
            mixture_prob += p
        if mixture_prob:
            Z[i] = (1/mixture_prob)*temp
        else:
            Z[i] = temp
    return Z

def update_parameters(X, Z, pi_list, mu_list, Sigma_list):
    """ Return the new parameters of the GMM model given Z

        X: (N,M) numpy ndarray
        Z: (N,K) numpy ndarray
        pi_list: List of K scalar values: the marginal probablity of each cluster
        mu_list: List of K (M, ) numpy arrays, mean of each cluster
        Sigma_list: List of K (M, M) numpy arrays, covariance matrix of each cluster
        
        Returns a tuple containing three lists: (new_pi_list, new_mu_list, new_Sigma_list)
            new_pi_list: List of K scalar values: the updated marginal probablity of each cluster
            new_mu_list: List of K (M, ) numpy arrays, updated mean of each cluster
            new_Sigma_list: List of K (M, M) numpy arrays, updated covariance matrix of each cluster
    """
    

    N, K = Z.shape
    M = X.shape[1]
    new_pi_list = (1/N)*Z.sum(axis=0)
    new_mu_list = ([0])*K
    new_Sigma_list = ([0])*K
    for i in range(K):
        z_col = Z[:, i]
        mu_temp_array = np.zeros((M, ))
        for j in range(N):
            sample = X[j]
            mu_temp_array = mu_temp_array + (z_col[j]*sample)
        den = np.sum(z_col, axis = 0)
        if den:
            new_mu_list[i] = (1/den) * mu_temp_array
        else:
            new_mu_list[i] = mu_temp_array
        
    for i in range(K):
        z_col = Z[:, i]
        Sigma_temp_mat = np.zeros((M, M))
        for j in range(N):
            sample = X[j]
            mult= np.matmul((sample - new_mu_list[i]).reshape(M,1), ((sample - new_mu_list[i]).reshape(M,1)).T)
            Sigma_temp_mat = Sigma_temp_mat +  (z_col[j] * mult)
        den = np.sum(z_col, axis = 0)
        if den == 0:
            new_Sigma_list[i] = Sigma_temp_mat
        else:
            new_Sigma_list[i] = (1/den) * Sigma_temp_mat
    
    return (new_pi_list, new_mu_list, new_Sigma_list)

def learn_gmm(X, pi_list, mu_list, Sigma_list, max_iters):
    """ Learn the GMM parameters by alternating between updating
        cluster probabilities and updateing the parameters

        X: (N,M) numpy ndarray
        pi_list: List of K scalar values: the marginal probablity of each cluster
        mu_list: List of K (M, ) numpy arrays, mean of each cluster
        Sigma_list: List of K (M, M) numpy arrays, covariance matrix of each cluster
        max_iters: int, number of iterations to run
        
        Returns a tuple containing three lists: (new_pi_list, new_mu_list, new_Sigma_list)
            new_pi_list: List of K scalar values: the updated marginal probablity of each cluster
            new_mu_list: List of K (M, ) numpy arrays, updated mean of each cluster
            new_Sigma_list: List of K (M, M) numpy arrays, updated covariance matrix of each cluster
    """
    N = X.shape[0]
    K = len(pi_list)

    Z = np.zeros((N, K))

    initial_likelihood = log_likelihood(X, pi_list, mu_list, Sigma_list)
    print("The initial log likelihood is {}".format(initial_likelihood))

    #util.plot_gmm(X, mu_list, Sigma_list)

    # Execute EM for GMM

    for i in range(max_iters):
        update_Z(X, Z, pi_list, mu_list, Sigma_list)

        #print('Z:')
        #print(Z)

        pi_list, mu_list, Sigma_list = update_parameters(X, Z, pi_list, mu_list, Sigma_list)

        #print('mus:')
        #for mu in mu_list:
        #    print(mu)

        #print('Sigmas:')
        #for Sigma in Sigma_list:
        #    print(Sigma)
    
        cur_likelihood = log_likelihood(X, pi_list, mu_list, Sigma_list)
        print("The log likelihood after iteration {} is {}".format(i+1, cur_likelihood))

        #util.plot_gmm(X, mu_list, Sigma_list)

    return (pi_list, mu_list, Sigma_list)

