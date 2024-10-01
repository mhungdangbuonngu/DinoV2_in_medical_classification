from torch.utils.data import DataLoader
import model as md
from data_loader import train_image_datasets
import json

batch_size = 32
num_workers = 4

data_loaders = {x: DataLoader(train_image_datasets[x], shuffle=True, batch_size=batch_size, num_workers=4)
    for x in ['train', 'test']
}

epoch_loss = md.train(data_loaders, iters=5)

with open(r"model_eval/overall_loss.json", "w") as f:
    json.dump(epoch_loss, f)