import numpy as np

def array_to_1(y):
    return [np.argmax(label) for label in y]


def load_score(y_true_np,y_pred_np):
    y_true=np.load(y_true_np)
    y_pred=np.load(y_pred_np)
    return y_true,y_pred


def accuracy(y_true, y_pred):
    y_true=array_to_1(y_true)
    y_pred=array_to_1(y_pred)
    correct_prediction = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct_prediction += 1
    accuracy = correct_prediction / len(y_true)
    return accuracy

def precision(y_true,y_pred):
    y_true=array_to_1(y_true)
    y_pred=array_to_1(y_pred)
    tp=0
    fp=0
    for i in range(len(y_true)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                tp+=1
            else:
                fp+=1
    if tp + fp == 0:
        return 0
    return tp/(tp+fp)

def recall(y_true,y_pred):
    y_true=array_to_1(y_true)
    y_pred=array_to_1(y_pred)
    tp=0
    fn=0
    for i in range(len(y_true)):
        if y_true[i] == 1:  
            if y_pred[i] == 1:
                tp += 1 
            else:
                fn += 1  
    
    if tp + fn == 0:
        return 0 
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    
    if (prec + rec) == 0:
        return 0
    
    return 2 * (prec * rec) / (prec + rec)