import numpy as np
import pickle


def _sigmoid(x):
    """Compute the sigmoid value of x."""
    return 1 / (1 + np.exp(-x))


def _get_grad(w, b, XTrain, yTrain):
    """
    Compute gradient.
    
    :param
        XTrain: numpy array of size [num_samples, feat_dim]
          where num_samples is the number of samples
          and feat_dim is the dimension of features
        yTrain: numpy array of size [num_samples, 1]
        w: weight vector of w
        b: bias

    :return: 
        w_grad: gradient for w
        b_grad: gradient for b    
    """
    n, d = XTrain.shape
    activations = _sigmoid(XTrain @ w + b)
    w_grad = (1/n) * XTrain.T @ (activations - yTrain)
    bias_grad = np.mean(activations - yTrain)
    return w_grad, bias_grad


def run(XTrain, yTrain, XTest, yTest, num_epochs, learning_rate):
    """
    Train logistic regression and return train and test accuracy.
    
    :param
        XTrain: numpy array of size [num_samples, feat_dim]
          where num_samples is the number of samples
          and feat_dim is the dimension of features
        yTrain: numpy array of size [num_samples, 1]
        XTest: numpy array of size [num_test, feat_dim]
          where num_test is the number of test samples
          and feat_dim is the dimension of features
        yTrain: numpy array of size [num_test, 1]
        num_epochs: number of training epochs
        learning_rate: learning rate

    :return
        train_acc: training accuracy
        test_acc: test accuracy
    """
    yTrain = 2 - yTrain
    yTest = 2 - yTest

    # Initialize w and b at 0.
    d = XTrain.shape[1]
    w = np.zeros((d,1))
    b = 0.0

    for _ in range(num_epochs):
        # Update step.
        grad_w, grad_b = _get_grad(w, b, XTrain, yTrain)
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

    # Train metrics.
    activations = _sigmoid(XTrain @ w + b)
    train_acc = np.mean((activations >= 0.5) == yTrain)
    
    # Test metrics.
    test_activations = _sigmoid(XTest @ w + b)
    test_acc = np.mean((test_activations >= 0.5) == yTest)
  
    return train_acc, test_acc

def dup_cols(a, indx, num_dups=1):
    return np.insert(a,[indx+1]*num_dups,a[:,[indx]],axis=1)

if __name__ == "__main__":
    lr1 = 0.1
    num_epochs1 = 5000
    dataFile = open("hw1data.pkl", 'rb')
    db = pickle.load(dataFile)
    XTrain = db["XTrain"]
    yTrain = db["yTrain"]
    XTest = db["XTest"]
    yTest = db["yTest"]
    XTrain = XTrain[:, : 10]
    XTest = XTest[:, : 10]
    XTrain = dup_cols(XTrain, 4, num_dups=500)
    XTest  = dup_cols(XTest, 4, num_dups=500)
    print(XTrain.shape)
    train_acc, test_acc = run(XTrain, yTrain, XTest, yTest, num_epochs1, lr1)
    print("training accuracy ", train_acc)
    print("testing accuracy ", test_acc) 