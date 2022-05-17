import numpy as np
import math

def load_data_small():
    """ Load small training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An (N_train, M) ndarray containing the training data (N_train examples, M features each)
        y_train: An (N_train,) ndarray contraining the labels
        X_val: An (N_val, M) ndarray containing the validation data (N_val examples, M features each)
        y_val: An (N_val,) ndarray contraining the labels
    """
    train_all = np.loadtxt('data/smallTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/smallValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_medium():
    """ Load medium training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An (N_train, M) ndarray containing the training data (N_train examples, M features each)
        y_train: An (N_train,) ndarray contraining the labels
        X_val: An (N_val, M) ndarray containing the validation data (N_val examples, M features each)
        y_val: An (N_val,) ndarray contraining the labels
    """
    train_all = np.loadtxt('data/mediumTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/mediumValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_large():
    """ Load large training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An (N_train, M) ndarray containing the training data (N_train examples, M features each)
        y_train: An (N_train,) ndarray contraining the labels
        X_val: An (N_val, M) ndarray containing the validation data (N_val examples, M features each)
        y_val: An (N_val,) ndarray contraining the labels
    """
    train_all = np.loadtxt('data/largeTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/largeValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def linearForward(input, p):
    """
    :param input: input vector (column vector) WITH bias feature added
    :param p: parameter matrix (alpha/beta) WITH bias parameter added
    :return: output vector
    """
    return np.array(np.matmul(p, input))

# custom function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def sigmoidForward(a):
    """
    :param a: input vector WITH bias feature added
    """
    # define vectorized sigmoid
    sigmoid_v = np.vectorize(sigmoid)
    return sigmoid_v(a)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    
def softmaxForward(b):
    """
    :param b: input vector WITH bias feature added
    """
    return softmax(b)

def CE(y,y_hat):
    return y * np.log(y_hat)

def crossEntropyForward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    :return: float
    """

    crossEntrop = np.vectorize(CE)
    J = -1 * float(np.sum(crossEntrop(hot_y, y_hat), axis = 0))
    return J


def NNForward(x, y, alpha, beta):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: beta WITH bias parameter added
    :return: all intermediate quantities x, a, z, b, y, J #refer to writeup for details
    TIP: Check on your dimensions. Did you make sure all bias features are added?
    """
    
    a = linearForward(x, alpha)
    z = sigmoidForward(a)
    z = np.insert(z, 0, 1, axis = 0)
    b = linearForward(z, beta)
    y_hat = softmaxForward(b)
    y_hot = np.zeros(y_hat.shape)
    y_hot[y] = 1
    J = crossEntropyForward(y_hot, y_hat)
    return x, a, z, b, y_hat, J

def softmaxBHelper(y_j, y_hat_j):
    softmaxB = y_hat_j - y_j
    return softmaxB
    
def softmaxBackward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    """
    softB = np.vectorize(softmaxBHelper)
    return softB(hot_y, y_hat)
    

def linearBackward(prev, p, grad_curr):
    """
    :param prev: previous layer WITH bias feature
    :param p: parameter matrix (alpha/beta) WITH bias parameter
    :param grad_curr: gradients for current layer
    :return:
        - grad_param: gradients for parameter matrix (alpha/beta)
        - grad_prevl: gradients for previous layer
    TIP: Check your dimensions
    """
    
    grad_param = np.matmul(grad_curr, prev.T)
    p = p[:,1:]
    grad_prevl = np.matmul(p.T, grad_curr)
    return grad_param, grad_prevl

def sigBackH(x):
    return x * (1.0 - x)

def sigmoidBackward(curr, grad_curr):
    """
    :param curr: current layer WITH bias feature
    :param grad_curr: gradients for current layer
    :return: grad_prevl: gradients for previous layer
    TIP: Check your dimensions
    """
    
    sigH = np.vectorize(sigBackH)
    return np.multiply(sigH(curr[1:, :]), grad_curr)


def NNBackward(x, y, alpha, beta, z, y_hat):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: alpha WITH bias parameter added
    :param z: z as per writeup
    :param y_hat: vector of probabilistic distribution for predicted label
    :return:
        - grad_alpha: gradients for alpha
        - grad_beta: gradients for beta
        - g_b: gradients for layer b (softmaxBackward)
        - g_z: gradients for layer z (linearBackward)
        - g_a: gradients for layer a (sigmoidBackward)
    TIP: Make sure you're accounting for the changes due to the bias term
    """
    
    y_hot = np.zeros(y_hat.shape)
    y_hot[y] = 1
    grad_b = softmaxBackward(y_hot, y_hat)
    grad_beta, grad_z = linearBackward(z, beta, grad_b)
    grad_a = sigmoidBackward(z, grad_z)
    grad_alpha, grad_x = linearBackward(x, alpha, grad_a)
    
    return grad_alpha, grad_beta, grad_b, grad_z, grad_a


def set_arr(rows, cols, init_rand):
    arr = np.zeros((rows, cols))
    if init_rand:
        for i in range(rows):
            for j in range(cols):
                if j != 0:
                    arr[i,j] = np.random.uniform(-0.1, 0.1)
    return arr

def SGD(X_train, y_train, X_val, y_val, hidden_units, num_epochs, init_rand, learning_rate):
    """
    :param X_train: Training data input (ndarray with shape (N_train, M))
    :param y_train: Training labels (1D column vector with shape (N_train,))
    :param X_val: Validation data input (ndarray with shape (N_valid, M))
    :param y_val: Validation labels (1D column vector with shape (N_valid,))
    :param hidden_units: Number of hidden units
    :param num_epochs: Number of epochs
    :param init_rand:
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
    :param learning_rate: Learning rate
    :return:
        - alpha weights
        - beta weights
        - train_entropy (length num_epochs): mean cross-entropy loss for training data for each epoch
        - valid_entropy (length num_epochs): mean cross-entropy loss for validation data for each epoch
    """
    N = X_train.shape[0]
    N_val = X_val.shape[0]
    alpha = set_arr(hidden_units, X_train.shape[1] + 1, init_rand)
    K = 10
    beta = set_arr(K, hidden_units + 1, init_rand)
    losses_train = np.array([])
    losses_val = np.array([])
    for epoch in range(num_epochs):
        for i in range(N):
            x = X_train[i]
            x = np.insert(x, 0, 1, axis = 0)
            x = np.array([x]).T
            y = y_train[i]
            x, a, z, b, y_hat, J = NNForward(x, y, alpha, beta)
            grad_alpha, grad_beta, grad_b, grad_z, grad_a = NNBackward(x, y, alpha, beta, z, y_hat)
            alpha = alpha - (learning_rate*grad_alpha)
            beta = beta - (learning_rate*grad_beta)
        
        J_arr = np.array([])
        for i in range(N):
            x = X_train[i]
            x = np.insert(x, 0, 1, axis = 0)
            x = np.array([x]).T
            y = y_train[i]
            x, a, b, z, y_hat, J = NNForward(x, y, alpha, beta)
            J_arr = np.append(J_arr, J)
        J_train = (1/N) * np.sum(J_arr)
        losses_train = np.append(losses_train, J_train)
        J_val_arr = np.array([])
        for j in range(N_val):
            x = X_val[j]
            x = np.insert(x, 0, 1, axis = 0)
            x = np.array([x]).T
            y = y_val[j]
            x, a, b, z, y_hat, J = NNForward(x, y, alpha, beta)
            J_val_arr = np.append(J_val_arr, J)
        J_val = (1/N_val) * np.sum(J_val_arr)
        losses_val = np.append(losses_val, J_val)
    return alpha, beta, losses_train, losses_val
    


def prediction(X_train, y_train, X_val, y_val, tr_alpha, tr_beta):
    """
    :param X_train: Training data input (ndarray with shape (N_train, M))
    :param y_train: Training labels (1D column vector with shape (N_train,))
    :param X_val: Validation data input (ndarray with shape (N_valid, M))
    :param y_val: Validation labels (1D column vector with shape (N_valid,))
    :param tr_alpha: Alpha weights WITH bias
    :param tr_beta: Beta weights WITH bias
    :return:
        - train_error: training error rate (float)
        - valid_error: validation error rate (float)
        - y_hat_train: predicted labels for training data (list)
        - y_hat_valid: predicted labels for validation data (list)
    """
    N_train = X_train.shape[0]
    corr_labels_train = 0
    y_hat_train = list()
    y_hat_valid = list()
    for i in range(N_train):
        x = X_train[i]
        y = y_train[i]
        x = np.insert(x, 0, 1, axis = 0)
        x, a, b, z, y_hat, J = NNForward(x.T, y, tr_alpha, tr_beta)
        l = np.argmax(y_hat)
        y_hat_train.append(l)
        if y == l:
            corr_labels_train += 1
    N_val = X_val.shape[0]
    corr_labels_val = 0
    for j in range(N_val):
        x = X_val[j]
        y = y_val[j]
        x = np.insert(x, 0, 1, axis = 0)
        x, a, b, z, y_hat, J = NNForward(x.T, y, tr_alpha, tr_beta)
        l = np.argmax(y_hat)
        y_hat_valid.append(l)
        if y == l:
            corr_labels_val += 1
    train_error = (1-float(corr_labels_train/N_train))
    valid_error =(1-float(corr_labels_val/N_val))
    return train_error, valid_error, y_hat_train, y_hat_valid
        

### FEEL FREE TO WRITE ANY HELPER FUNCTIONS

def train_and_valid(X_train, y_train, X_val, y_val, num_epochs, num_hidden, init_rand, learning_rate):
    """ 
    Main function to train and validate your neural network implementation.

    :param X_train: Training data input (ndarray with shape (N_train, M))
    :param y_train: Training labels (1D column vector with shape (N_train,))
    :param X_val: Validation data input (ndarray with shape (N_valid, M))
    :param y_val: Validation labels (1D column vector with shape (N_valid,))
    :param num_epochs: Number of epochs to train (i.e. number of loops through the training data).
    :param num_hidden: Number of hidden units.
    :param init_rand: Boolean value of True/False
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
    :param learning_rate: Float value specifying the learning rate for SGD.

    :return: a tuple of the following six objects, in order:
        - loss_per_epoch_train (length num_epochs): A list of float values containing the mean cross entropy on training data after each SGD epoch
        - loss_per_epoch_val (length num_epochs): A list of float values containing the mean cross entropy on validation data after each SGD epoch
        - err_train: Float value containing the training error after training (equivalent to 1.0 - accuracy rate)
        - err_val: Float value containing the validation error after training (equivalent to 1.0 - accuracy rate)
        - y_hat_train: A list of integers representing the predicted labels for training data
        - y_hat_val: A list of integers representing the predicted labels for validation data
    """
    alpha, beta, loss_per_epoch_train, loss_per_epoch_val = SGD(X_train, y_train, X_val, y_val, num_hidden, num_epochs, init_rand, learning_rate)
    err_train, err_val, y_hat_train, y_hat_val = prediction(X_train, y_train, X_val, y_val, alpha, beta)
    
    return (loss_per_epoch_train, loss_per_epoch_val,
            err_train, err_val, y_hat_train, y_hat_val)
