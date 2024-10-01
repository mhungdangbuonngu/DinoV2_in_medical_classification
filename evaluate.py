import torch
from torch.utils.data import DataLoader
from data_loader import test_image_datasets
from model import get_available_device, BrainTumorClassifier
import numpy as np
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')

device = get_available_device()

def give_predict(model_path, data, save_to_file=False) -> float:
    tmp_model = BrainTumorClassifier()
    tmp_model.load_state_dict(torch.load(model_path, weights_only=True))
    tmp_model.eval()
    
    tmp_model.to(device)
    print(f"using: {device}")
    loop = tqdm(data['test'])
    
    omg_predict = []
    omg_truth = []
    with torch.no_grad():
        for input_, truth_ in loop:
            input_ = input_.to(device)
            output_ = tmp_model(input_).squeeze(1)

            # output_ = output_.squeeze(1) 
            omg_predict.append(output_.detach().cpu().numpy())
            omg_truth.append(truth_)

    print(f'{len(omg_predict)} - {omg_predict[0].shape}')
    print(f'{len(omg_truth)} - {omg_truth[0].shape}')

    if save_to_file:
        np.save(r"model_eval/omg_pred", np.array(omg_predict[:-1]))
        np.save(r"model_eval/omg_truth", np.array(omg_truth[:-1]))
    
    return np.array(omg_predict[:-1]), np.array(omg_truth[:-1])

if __name__ == "__main__":
    batch_size = 32
    num_workers = 4

    data_loaders = {x: DataLoader(test_image_datasets[x], shuffle=True, batch_size=batch_size, num_workers=4)
        for x in ['train', 'test']
    }

    y_hat , y_truth = give_predict("model_eval/classifier_weight.pth", data_loaders, save_to_file=True)