import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset


def get_mnist(data_dir='./data/mnist/',batch_size=128):
    train = MNIST(root=data_dir, train=True, download=True)
    test = MNIST(root=data_dir, train=False, download=True)

    x = torch.cat([train.data.float().view(-1,784)/255., test.data.float().view(-1,784)/255.], 0)
    y = torch.cat([train.targets, test.targets], 0)

    dataset = dict()
    dataset['x'] = x
    dataset['y'] = y

    dataloader=DataLoader(TensorDataset(x,y), batch_size=batch_size, 
                          shuffle=True, num_workers=4)
    return dataloader