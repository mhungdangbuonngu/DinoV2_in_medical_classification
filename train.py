from data_loader import ChestXRDataset
from torch.utils.data import DataLoader
import model as md
import numpy as np

# train_label_path = "/mnt/g/Code/Dataset/archive/train/class_label"
# train_data_path = "/mnt/g/Code/Dataset/archive/train/img_feature"
# train_data = ChestXRDataset(train_data_path, train_label_path)

# train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

test_data_path = "/mnt/g/Code/Dataset/archive/test/img_feature"
test_label_path = "/mnt/g/Code/Dataset/archive/test/class_label"
test_data = ChestXRDataset(test_data_path, test_label_path)

test_data_loader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=4)

epoch_loss , batch_loss = md.train(test_data_loader, iters=5)

np.save("model_eval/epoch_loss",np.array(epoch_loss))
np.save("model_eval/batch_loss", np.array(batch_loss))