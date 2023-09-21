import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

def get_toy_data(batch_size):
    # Load the data from the CSV file
    df = pd.read_csv('toy_data.csv')

    # Convert the DataFrame to PyTorch tensors
    H = torch.tensor(df['H'].values, dtype=torch.long)  # H is a categorical variable
    L = torch.tensor(df['L'].values, dtype=torch.long)  # L is a categorical variable
    HL = torch.tensor(df['HL'].values, dtype=torch.long)  # L is a categorical variable    
    X1 = torch.tensor(df['X1'].values, dtype=torch.float32)
    X2 = torch.tensor(df['X2'].values, dtype=torch.float32)
    Y = torch.tensor(df['Y'].values, dtype=torch.float32)

    # Combine X1, X2, Y into a single tensor
    X = torch.stack([X1, X2, Y], dim=1)

    # Combine the tensors into a single dataset
    dataset = TensorDataset(X, HL)

    # Create a DataLoader from the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader
