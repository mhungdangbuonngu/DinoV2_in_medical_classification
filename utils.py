import numpy as np


def accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

def hamming_loss(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
    return temp/(y_true.shape[0] * y_true.shape[1])


def recall(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_true[i])
    return temp / y_true.shape[0]


def precision(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    return temp / y_true.shape[0]


def f1_measure(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        temp += (2 * sum(np.logical_and(y_true[i], y_pred[i]))) / (sum(y_true[i]) + sum(y_pred[i]))
    return temp / y_true.shape[0]
