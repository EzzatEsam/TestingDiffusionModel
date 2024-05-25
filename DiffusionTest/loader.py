import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the path to your dataset
data_dir = r"/teamspace/studios/this_studio/datasets/celeba_hq_root"

# Define the data transforms
transform = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]
)

# Create the dataset and data loader
dataset = datasets.ImageFolder(data_dir, transform=transform)

train , val = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train, batch_size=64, shuffle=True)
val_loader = DataLoader(val, batch_size=64, shuffle=False)
