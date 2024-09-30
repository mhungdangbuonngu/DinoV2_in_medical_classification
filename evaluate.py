import torch
from torch.utils.data import DataLoader
from data_loader import ChestXRDataset
from model import get_available_device, ChestXRClassifier
import numpy as np
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')

device = get_available_device()

def IoU_accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

def Hamming_Loss(y_true, y_pred):
    temp=0
    for i in range(y_true.shape[0]):
        temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
    return temp/(y_true.shape[0] * y_true.shape[1])

def Recall(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_true[i])
    return temp/ y_true.shape[0]

def Precision(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_pred[i])
    return temp/ y_true.shape[0]

def F1Measure(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        temp+= (2*sum(np.logical_and(y_true[i], y_pred[i])))/ (sum(y_true[i])+sum(y_pred[i]))
    return temp/ y_true.shape[0]

def accuracy(model_path, data) -> float:
    tmp_model = ChestXRClassifier()
    tmp_model.load_state_dict(torch.load(model_path, weights_only=True))
    tmp_model.eval()
    
    tmp_model.to(device)
    print(f"using: {device}")
    loop = tqdm(data)
    
    omg_predict = []
    omg_truth = []

    for input_, truth_ in loop:
        input_ = input_.to(device)
        output_ = tmp_model(input_)

        output_ = output_.squeeze(1) 
        omg_predict.append(output_.detach().cpu().numpy())
        omg_truth.append(truth_)

    print(f'{len(omg_predict)} - {omg_predict[0].shape}')
    print(f'{len(omg_truth)} - {omg_truth[0].shape}')

    np.save(r"model_eval/omg_pred", np.array(omg_predict[:-1]))
    np.save(r"model_eval/omg_truth", np.array(omg_truth[:-1]))

if __name__ == "__main__":
    test_data_path = r"npy_data/test/img_feature"
    test_label_path = r"npy_data/test/class_label"
    test_data = ChestXRDataset(test_data_path, test_label_path)

    test_data_loader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=6)

    accuracy("model_eval/classifier_weight.pth", test_data_loader)