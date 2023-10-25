import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

def get_toy_data(batch_size):
    # Load the data from the CSV file
    df = pd.read_csv('toy_data.csv')
    
    # Convert the DataFrame to PyTorch tensors
    H = torch.tensor(df['H'].values, dtype=torch.long)  # H is a categorical variable
    L = torch.tensor(df['L'].values, dtype=torch.long)  # L is a categorical variable
    HL = torch.tensor(df['HL'].values, dtype=torch.long)  # L is a categorical variable
    
    # Create a list to hold the X tensors
    X_tensors = []
    
    for col in df.columns:
        if col.startswith('X'):
            X_tensors.append(torch.tensor(df[col].values, dtype=torch.float32))
    
    Y = torch.tensor(df['Y'].values, dtype=torch.float32)
    
    # Combine the X tensors and Y into a single tensor
    X = torch.stack(X_tensors + [Y], dim=1)
    
    # scale observed X here, and save scaler
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(X)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # plt.scatter(X[:,0], scaled_data[:,0])
    
    # Combine the tensors into a single dataset
    dataset = TensorDataset(torch.tensor(scaled_data).float(), HL)
    
    # Define the size of the train and test datasets
    train_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size

    # Create the train and test datasets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Now you can create data loaders for the train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


