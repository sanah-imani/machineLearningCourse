import numpy as np

def load_data():
    """ Load randomly permuted MNIST dataset

        Returns an 30000-by-784 numpy ndarray X_train
    """

    X = np.load('data/mnist.npy').astype(np.float32)
    X_train = np.reshape(X[:30000][:][:], (30000, 784))
    return X_train

def kmeans_loss(X, C, z):
    """ Compute the K-means loss.

        Input:
        X: a numpy ndarray with shape (N,M), where each row is a data point
        C: a numpy ndarray with shape (K,M), where each row is a cluster center
        z: a numpy ndarray with shape (N,) where the i-th entry is an int from {0..K-1}
            representing the cluster index for the i-th point in X

        Returns mean squared distance from each point to the center for its assigned cluster
    """
    N = X.shape[0]
    loss = 0
    for i in range(N):
        diff = X[i] - C[z[i]]
        loss += diff @ diff / N

    return loss


# Feel free to add any helper functions you need here

def update_clusters(X, z, K):
    N, M = X.shape
    new_cent = []
    for idx in range(K):
        temp_arr = X[z==idx].mean(axis=0) 
        new_cent.append(temp_arr)
    return np.array(new_cent)
    



def kmeans(X, K):
    """ Cluster data X into K converged clusters.
    
        X: an N-by-M numpy ndarray, where we want to assign each
            of the N data points to a cluster.

        K: an integer denoting the number of clusters.

        Returns a tuple of length two containing (C, z):
            C: a numpy ndarray with shape (K,M), where each row is a cluster center
            z: a numpy ndarray with shape (N,) where the i-th entry is an int from {0..K-1}
                representing the cluster index for the i-th point in X
    """
    N = X.shape[0]
    M = X.shape[1]

    # Initialize cluster centers to the first K points of X
    C = np.copy(X[:K])

    # Initialize z temporarily to all -1 values
    z = -1*np.ones(N, dtype=np.int)

    converged = -1
    
    while converged:
        # for each datapoint in dataset
        for ind, row in enumerate(X):
            #finding min - use big distance first
            min_dist = float('inf')
            # calculating distance of the point from all centers
            for j, cent in enumerate(C):
               #using the norm 
               d = np.linalg.norm(cent-row)
               # store closest centroid
               if min_dist > d:
                  min_dist = d
                  z[ind] = j
        #new_C = pd.DataFrame(X).groupby(by=z).mean().values
        new_C = update_clusters(X, z, K)
        # if centroids are same then break out -> convergence
        if np.count_nonzero(C-new_C):
           C = new_C
        else:
            converged = 0
    return (C, z)

