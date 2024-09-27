import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class ChestXRClassifier(nn.Module):
    def __init__(self):
        super(ChestXRClassifier, self).__init__()

        #layer
        self.linear_layer1 = nn.Linear(384, 256) # input of size 256
        self.relu_layer = nn.ReLU()
        self.linear_layer2 = nn.Linear(256, 15)  # output of size 15
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        z = self.linear_layer1(x)
        z = self.relu_layer(z)
        z = self.linear_layer2(z)
        z = self.sigmoid_layer(z)

        return z
    
def get_available_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def train(data_loaders, iters=10, learning_rate=0.000001, wsp="$home/classifier_weight.pth"):
    device = get_available_device()
    model = ChestXRClassifier()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(iters):
        data_loop = tqdm(data_loaders['train'])