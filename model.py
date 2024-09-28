import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class ChestXRClassifier(nn.Module):
    def __init__(self):
        super(ChestXRClassifier, self).__init__()

        #layers
        self.linear_layer1 = nn.Linear(387, 256) # input of size 256
        self.relu_layer = nn.ReLU()
        self.linear_layer2 = nn.Linear(256, 15)  # output of size 15

    def forward(self, x):
        z = self.linear_layer1(x)
        z = self.relu_layer(z)
        z = self.linear_layer2(z)

        return z
    
def get_available_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def train(data_loaders, iters=10, learning_rate=0.000001, wsp="model_eval/classifier_weight.pth"):
    device = get_available_device()
    model = ChestXRClassifier()
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    for epoch in range(iters):
        loop = tqdm(data_loaders)
        
        epochs_loss = []
        batch_loss = []

        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs).squeeze(1)  # Forward pass

            loss = criterion(outputs, labels)  # Compute loss
            batch_loss.append(loss.item())
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            loop.set_description(f"Epoch [{epoch}/{iters}]")
            loop.set_postfix(loss=loss.item())
        
        epochs_loss.append(loss.item())

    torch.save(model.state_dict(), wsp)

    return epochs_loss, batch_loss