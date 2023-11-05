import numpy as np 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# MODEL ARCHITECTURE
class Model(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cnn = torchvision.models.resnet18(pretrained = False, 
                                               num_classes = 4 * hidden_dim)
        self.mlp = nn.Sequential(nn.ReLU(inplace = True),
                                   nn.Linear(4 * hidden_dim, hidden_dim)
                                   )
    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x)
        return x
    
# TRAIN FUNCTION
def train(dataloader, model, epochs, learning_rate):
    model = model.to(device)
    model.train()
    loss_fnc = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    
    losses = []
    for epoch in range(epochs):
        running_loss = 0
        for batch, (images, labels) in enumerate(dataloader):
            # Forward pass
            output = model(images.to(device))
            # Compute ;pss
            loss = loss_fnc(output, labels.to(device))
            
            # Zero the gradients before backpropagation
            optimizer.zero_grad()
            # Backpropagation (backward pass)
            loss.backward()
            # Update parameters of a model based on gradients
            # calculated during backprop
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / (batch + 1)
        losses.append(avg_loss)
        print(f'Training loss {epoch+1} = {avg_loss:.7f}')
    print('\nTraining complete! (wohoo!)')
    return losses

# TEST FUNCTION
def test(dataloader, model):
    model = model.to(device)
    model.eval()
    loss_fnc = nn.CrossEntropyLoss()
    
    total = 0
    correct = 0
    
    for batch, (images,labels) in enumerate(dataloader):
        # Forward pass
        output = model(images.to(device))
        
        total += labels.size(0)
        correct += (output.argmax(dim=1) == labels.to(device)).sum().item()
    
    print(f'Accuracy on Test set = {100 * (correct/total):.3f}% [{correct}\{total}]')

# Finetune pretrained SimCLR model
# Load the saved encoder
model_path = "/home/ubuntu/code/simCLR_ep500.pth"
simclr = Model(128)
simclr.load_state_dict(torch.load(model_path))
#simclr = torch.load(model_path)

simclr.cnn.fc = nn.Linear(512, 10)
simclr.mlp = nn.Identity()
print(simclr)

train_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,),(0.5,))
                                     ])

train_data = datasets.STL10(root='./data',split='train',download=True,transform=train_transforms)
train_dataloader = DataLoader(train_data,batch_size=32,shuffle=True,pin_memory=True)

simclr_losses = train(train_dataloader,simclr, epochs=200, learning_rate=1e-3)

baseline = models.resnet18(pretrained=False,num_classes=10)


train_transforms = transforms.Compose([
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(size=96, scale=(0.8, 1.0)),
                                       transforms.RandomGrayscale(p=0.2),
                                       transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 0.5)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,))
]) #To match the data used to pretrain.

train_data = datasets.STL10(root='./data',split='train',download=True,transform=train_transforms)
train_dataloader = DataLoader(train_data,batch_size=32,shuffle=True,pin_memory=True)

baseline_losses = train(train_dataloader, baseline, epochs=200, learning_rate=1e-3)

# Plot of SimCLR & Baseline's losses during Training
plt.figure(figsize=(10, 5))
plt.title("SimCLR & Baseline Training Loss")
plt.plot(simclr_losses, label="SimCLR")
plt.plot(baseline_losses, label="Baseline")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot.png")  # Save the figure as an image (e.g., loss_plot.png)
plt.show()  # Optionally, display the figure (you can remove this line if you don't want to display it)



test_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,),(0.5,))
])

test_data = datasets.STL10(root='./data',split='test',download=True,transform=test_transforms)
test_dataloader = DataLoader(test_data,batch_size=32,shuffle=False,pin_memory=True,num_workers=1)


print('For Neural Network pretrained using SimCLR:')
test(test_dataloader,simclr)
print('\nFor Baseline:')
test(test_dataloader,baseline)
