import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json

class ChestXRDataset(Dataset):
    def __init__(self, data_path, label_path): #get all .npy file from directory
        self.data_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
        self.label_files = [os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith('.npy')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx): #get the data and label based on index
        data_sample = np.load(self.data_files[idx])
        label_sample = np.load(self.label_files[idx])
        
        #convert to tensors
        return torch.tensor(data_sample, dtype=torch.float32), torch.tensor(label_sample, dtype=torch.float32)