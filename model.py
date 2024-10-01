import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class BrainTumorClassifier(nn.Module):
    def __init__(self):
        super(BrainTumorClassifier, self).__init__()

        self.transformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.classifier = nn.Sequential(
            nn.Linear(384, 256), # input of size 384
            nn.ReLU(),
            nn.Linear(256, 4)
            )

    def forward(self, x):
        z = self.transformer(x)
        z = self.transformer.norm(z)
        z = self.classifier(z)

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
    print(f"Using: {device}")

    model = BrainTumorClassifier()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epochs_loss = {}

    for epoch in range(iters):
        loop = tqdm(data_loaders['train'])
        
        batch_loss = []

        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs).squeeze(1)  # Forward pass

            loss = criterion(outputs, labels)  # Compute loss
            batch_loss.append(loss.item())

            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == labels).sum().item()
            accuracy = correct / 32

            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            loop.set_description(f"Epoch [{epoch}/{iters}]")
            loop.set_postfix(loss=loss.item(), acc=accuracy)
        
        epochs_loss[epoch] = batch_loss

    torch.save(model.state_dict(), wsp)

    return epochs_loss