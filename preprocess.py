import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA

from global_settings import *

def get_mnist(data_dir='./data/mnist/',batch_size=128):
    train = MNIST(root=data_dir, train=True, download=True)
    test = MNIST(root=data_dir, train=False, download=True)

    x = torch.cat([train.data.float().view(-1,784)/255., test.data.float().view(-1,784)/255.], 0)
    y = torch.cat([train.targets, test.targets], 0)


    pca = PCA(n_components=in_dim)
    x2 = pca.fit_transform(x)
    x2_min = x2.min()
    x2_max = x2.max()
    x2_scaled = (x2 - x2_min) / (x2_max - x2_min)
    
    x = torch.from_numpy(x2_scaled).float()
    # pca.explained_variance_ratio_.sum()

    # dataset = dict()
    # dataset['x'] = x
    # dataset['y'] = y

    dataloader=DataLoader(TensorDataset(x,y), batch_size=batch_size, 
                          shuffle=True, num_workers=4)
    return dataloader