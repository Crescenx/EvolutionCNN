import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import DataLoaderConfig

transform = transforms.Compose([
    transforms.ToTensor(),
])

def load_data():
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    batch_size = DataLoaderConfig.BATCH_SIZE

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
