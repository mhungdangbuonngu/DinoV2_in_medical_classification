import torch
from torch.utils.data import DataLoader
from data_loader import ChestXRDataset
from model import get_available_device, ChestXRClassifier
import numpy as np
from tqdm import tqdm
import accuracy as acdc

torch.multiprocessing.set_sharing_strategy('file_system')

device = get_available_device()

def give_predict(model_path, data, save_to_file=False) -> float:
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

    if save_to_file:
        np.save(r"model_eval/omg_pred", np.array(omg_predict[:-1]))
        np.save(r"model_eval/omg_truth", np.array(omg_truth[:-1]))
    
    return np.array(omg_predict[:-1]), np.array(omg_truth[:-1])

if __name__ == "__main__":
    test_data_path = r"npy_data/test/img_feature"
    test_label_path = r"npy_data/test/class_label"
    test_data = ChestXRDataset(test_data_path, test_label_path)

    test_data_loader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=6)

    y_hat , y_truth = give_predict("model_eval/classifier_weight.pth", test_data_loader, save_to_file=False)

    y_hat = y_hat.reshape(32,2)
    y_truth = y_truth.reshape(32,2)

    print(f"Accuracy: {acdc.accuracy(y_truth, y_hat)}")
    print(f"Recall: {acdc.recall(y_truth, y_hat)}")
    print(f"Precision: {acdc.precision(y_truth, y_hat)}")
    print(f"F1 Score: {acdc.f1_score(y_truth, y_hat)}")