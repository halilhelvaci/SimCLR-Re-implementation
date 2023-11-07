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
print('Current device is: ', device)

# Transformations

class Transformations():
    def __init__(self, num_of_views=2):
        self.transforms = transforms.Compose([
            # Randomly flip the image horizontally (left to right)
            transforms.RandomHorizontalFlip(),
            # Randomly crop and resiz the image to the specified size (96x96 pixels)
            transforms.RandomResizedCrop(96),
            # Randomly apply the given transformations (in this case color jitter) with a probability of 0.8
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5,
                                                        contrast=0.5,
                                                        saturation=0.5,
                                                        hue=0.1)], p=0.8),

            # Randomly convert the image to grayscale with a probability of 0.2
            transforms.RandomGrayscale(p=0.2),
            # Apply Gaussian blur with the specified kernel size (9x9)
            transforms.GaussianBlur(kernel_size=9),
            # Convert the image to PyTorch tensor
            transforms.ToTensor(),
            # Normalize pixel values of the tensor (mean, std_dev) centered at 0 (±) 0.5
            # aka a range of 0 to 1
            transforms.Normalize((0.5,), (0.5,))
            ])
        self.num_of_views = num_of_views

    # Apply the transformations to an input data sample (an image).
    # It takes one argument, x, the input data. It applies the defined transformations
    # num_of_views times, creating a list of augmented views of the input data.
    def __call__(self, x):
        return [self.transforms(x) for i in range(self.num_of_views)]

unlabeled_data = STL10(root='./data', split='train', download=True, transform=Transformations())
unlabeled_dataloader = DataLoader(unlabeled_data, batch_size=800, shuffle=True, pin_memory=True)

# ML MODEL
class Model(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.cnn = resnet18(pretrained=False, num_classes = 4*hidden_dim)
        self.mlp = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim),)

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x)
        return x

model = Model(hidden_dim=128)

# TRAINING 
class SimCLR():
    def __init__(self, dataloader, model, temperature=0.07):
        self.dataloader = dataloader
        self.model = model.to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        self.temperature = temperature

    def train(self, num_epochs):
        self.model.train()

        for epoch in range(num_epochs):
            for batches in range(len(self.dataloader)): # looping over the batches
                batch = next(iter(self.dataloader))# take a single batch
                imgs = batch[0]
                imgs = torch.cat(imgs, dim=0)

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

simclr = SimCLR(unlabeled_dataloader, model)
simclr.train(num_epochs=1000)

model_path = ("/home/ubuntu/code/simclr_model.pth")
#torch.save(simclr.model, model_path+"model.pth")
torch.save(model.state_dict(), model_path)
