import torch
import torch.nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

params = {
    'epoch': 100,
    'batch_size': 64
}


def load_data():
    train_data = datasets.MNIST(root='..\\data\\MNIST', train=True, download=True, transform=None)
    test_data = datasets.MNIST(root='..\\data\\MNIST', train=False, download=True, transform=None)
    train_data = DataLoader(train_data, batch_size=params.get('batch_size'), shuffle=True)
    test_data = DataLoader(test_data, batch_size=params.get('batch_size'), shuffle=True)
    return train_data, test_data


def train():
    train_data, test_data = load_data()
    for i in range(len(test_data)):
        pass


if __name__ == '__main__':
    train()