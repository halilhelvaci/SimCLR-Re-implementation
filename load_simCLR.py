# essentials
import os
import numpy as np
import matplotlib.pyplot as plt
import random
# torch related
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# torch_vision related
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torchvision.models import resnet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ML MODEL
class Model(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.cnn = resnet18(pretrained = False, num_classes = 4 * hidden_dim)
        self.mlp = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.Linear(4 * hidden_dim, hidden_dim),)

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x)
        return x
    
# TRAINING 
class SimCLR():
    def __init__(self, dataloader, model, temperature = 0.07):
        self.dataloader = dataloader
        self.model = model.to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr = 5e-4, weight_decay = 1e-4)
        self.temperature = temperature

    def train(self, num_epochs):
        self.model.train()

        for epoch in range(num_epochs):
            for batches in range(len(self.dataloader)): # looping over the batches
                batch = next(iter(self.dataloader))# take a single batch
                imgs = batch[0]
                imgs = torch.cat(imgs, dim = 0)

                # Encode all images
                feats = self.model(imgs.to(device))
                # Calculate cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(feats[:, None, :], feats[None, :, :], dim = -1)
                # Mask out cosine similarity with itself
                self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device = cos_sim.device)
                cos_sim.masked_fill(self_mask, -9e15)
                # Find positive example -> batch_size//2 away from original example
                pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
                # InfoNCE loss
                cos_sim = cos_sim / self.temperature
                nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
                loss = nll.mean()

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch+1} - Loss:{loss:.7f}')

# Load the saved encoder
model_path = "/home/ubuntu/code/simCLR_ep200.pth"

model = Model(128)
model.load_state_dict(torch.load(model_path))

test_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,),(0.5,))
])

test_data = STL10(root='./data', split='test', download=True, transform=test_transforms)
test_dataloader = DataLoader(test_data, batch_size=800, shuffle=False, pin_memory=True, num_workers=2)

def test(dataloader,model):
    model = model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    for batch,(imgs,labels) in enumerate(dataloader):
        output = model(imgs.to(device))

        total += labels.size(0)
        correct += (output.argmax(dim=1) == labels.to(device)).sum().item()

    print(f'Accuracy on Testing set = {100 * (correct/total):.3f}% [{correct}/{total}]')
    
print('For Neural Network pretrained using SimCLR:')
test(test_dataloader, model)