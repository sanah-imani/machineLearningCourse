
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import pickle

def NB_XGivenY(XTrain, yTrain, a=0.001, b=0.9):
    """
    Compute the probability of P(X|Y).

    :param
        XTrain: numpy array of size [num_samples, feat_dim]
          where num_samples is the number of samples
          and feat_dim is the dimension of features
        yTrain: numpy array of size [num_samples, 1]
        a: default to 0.001
        b: default to 0.9

    :return: 
        D: numpy array of size [2, vocab_size] where
          vocab_size is the size of vocabulary
    """
    V = len(XTrain[0])
    N = len(XTrain)
    D = np.zeros((2, V))
    for i in range(len(D)):
        y_i = i + 1
        for j in range(V):
            count = 0
            count_y = 0
            for n_i in range(N):   
                 doc_label = yTrain[n_i]
                 if doc_label == y_i:
                     count_y += 1
                     if XTrain[n_i][j] == 1:
                         count += 1
            likelihood = (count + a) / (count_y + a + b)
            D[i,j] = likelihood
    D = np.clip(D, a_min = (10**(-5)), a_max = (1- 10**(-5)))
    return D


def NB_YPrior(yTrain):
    """
    Compute the probability of P(Y).

    :param
        yTrain: numpy array of size [num_samples, 1]

    :return: 
        p: a scalar for the probability of P(Y = 1)
    """
    n = len(yTrain)
    count = 0
    for i in range(n):
        if yTrain[i] == 1:
            count += 1
    p = count/n
    return p


def NB_Classify(D, p, X):
    """
    Predict the labels of X.

    :param
        D: the probability P(X|Y)
        p: the probability P(Y)
        X: numpy array of size [num_samples, feat_dim]
          where num_samples is the number of samples
          and feat_dim is the dimension of features

    :return: 
        y: numpy array of size [num_samples, 1] where
            num_samples is the number of samples
    """
    m = len(X)
    V = len(X[0])
    p_label1 = p
    p_label2 = 1 - p
    y = np.zeros((m, 1))
    for sample_id in range(m):
        argmax_label = 0
        maxProb = float("-inf")
        for y_label in range(2):
            p_y = 0
            curr_prob = 0
            if y_label + 1 == 1:
                p_y = p_label1
            else:
                p_y = p_label2
            for feature_i in range(V):
                
                p_xy = D[y_label][feature_i]
                
                if X[sample_id][feature_i]:
                    curr_prob += np.log(p_xy) 
                else:
                    curr_prob += np.log((1-p_xy)) 
            curr_prob += np.log(p_y)
            if math.isclose(maxProb, curr_prob) or maxProb < curr_prob:
                maxProb = curr_prob
                argmax_label = y_label + 1
        y[sample_id, 0] = argmax_label
    return y


def NB_ClassificationAccuracy(yHat, yTruth):
    """
    Compute the accuracy of predictions.

    :param
        yHat: numpy array of size [num_samples, 1]
        yTruth: numpy array of size [num_samples, 1]
    
    :return:
        acc: a scalar for the accuracy
    """
    
    num_samples = len(yHat)
    match_count = 0
    for sample_i in range(num_samples):
        if yHat[sample_i] == yTruth[sample_i]:
            match_count += 1
    acc = match_count/num_samples
    return acc

def dup_cols(a, indx, num_dups=1):
    return np.insert(a,[indx+1]*num_dups,a[:,[indx]],axis=1)

if __name__ == "__main__":
    dataFile = open("hw1data.pkl", 'rb')
    db = pickle.load(dataFile)
    XTrain = db["XTrain"]
    yTrain = db["yTrain"]
    XTest = db["XTest"]
    yTest = db["yTest"]
    XTrain = XTrain[:, : 10]
    XTest = XTest[:, : 10]
    #XTrain = dup_cols(XTrain, 4, num_dups=500)
    #XTest  = dup_cols(XTest, 4, num_dups=500)
    print(XTrain.shape)
    print(XTest.shape)
    D = NB_XGivenY(XTrain, yTrain)
    p = NB_YPrior(yTrain)
    
    yHatTrain = NB_Classify(D, p, XTrain)
    yHatTest = NB_Classify(D, p, XTest)

    trainAcc = NB_ClassificationAccuracy(yHatTrain, yTrain)
    testAcc = NB_ClassificationAccuracy(yHatTest, yTest)

    print("training accuracy ", trainAcc)
    print("testing accuracy ", testAcc)
    

