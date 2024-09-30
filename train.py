from data_loader import ChestXRDataset
from torch.utils.data import DataLoader
import model as md
import numpy as np
import json

train_label_path = r"npy_data/train/class_label"
train_data_path = r"npy_data/train/img_feature"
train_data = ChestXRDataset(train_data_path, train_label_path)

train_data_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)

epoch_loss = md.train(train_data_loader, iters=10)

with open(r"model_eval/overall_loss.json", "w") as f:
    json.dump(epoch_loss, f)