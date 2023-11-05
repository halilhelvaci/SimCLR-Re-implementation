import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
# torch_vision related
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torchvision.models import resnet18


# Define a function to extract feature vectors from the model
def get_features(model, data_loader, device):
    model.eval()
    feature_vectors = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            features = model(inputs)
            feature_vectors.extend(features.cpu().numpy())

    return np.array(feature_vectors)

# Define a function to perform t-SNE and visualize the results
def visualize_tSNE(features, labels, n_classes, output_image):
    tsne = TSNE(n_components=2, random_state=0)
    reduced_features = tsne.fit_transform(features)

    # Create a scatter plot for t-SNE visualization
    plt.figure(figsize=(8, 8))
    for i in range(n_classes):
        plt.scatter(reduced_features[labels == i, 0], reduced_features[labels == i, 1], label=f'Class {i}', s=10)
    plt.title('t-SNE Visualization')
    plt.legend()
    plt.show()
    plt.savefig(output_image, format='jpeg', dpi=300, bbox_inches='tight')

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


test_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,),(0.5,))
])

# Specify the path to your trained SimCLR model
model_path = "/home/ubuntu/code/simCLR_ep50.pth"
output_image = 'tSNE_ep50.jpg'

# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained SimCLR model
model = Model(128)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# Create a data loader for the STL-10 test set
#test_dataset = STL10(root=test_dataset_path, split='test', transform=transforms.ToTensor(), download=False)
test_dataset = STL10(root='./test_data', split='test', download=True, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=800, shuffle=False, pin_memory=True, num_workers=2)

# Get feature vectors for the test set
feature_vectors = get_features(model, test_loader, device)

# Get labels for the test set
labels = np.array(test_dataset.labels)

# Number of classes in the STL-10 dataset
n_classes = len(set(labels))

# Perform t-SNE and visualize the results
visualize_tSNE(feature_vectors, labels, n_classes, output_image)
