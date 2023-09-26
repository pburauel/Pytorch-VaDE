import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from sklearn.mixture import GaussianMixture

from global_settings import *



class VaDE(torch.nn.Module):
    def __init__(self, in_dim=in_dim, latent_dim=latent_dim, n_classes=n_classes): # latent_dim used to be 10
        super(VaDE, self).__init__()

        self.pi_prior = Parameter(torch.ones(n_classes)/n_classes)
        self.mu_prior = Parameter(torch.zeros(n_classes, latent_dim))
        self.log_var_prior = Parameter(torch.randn(n_classes, latent_dim))
        
        self.fc1 = nn.Linear(in_dim, encoder_units[0]) #Encoder
        self.fc2 = nn.Linear(encoder_units[0], encoder_units[1])
        self.fc3 = nn.Linear(encoder_units[1], encoder_units[2]) 

        self.mu = nn.Linear(encoder_units[2], latent_dim) #Latent mu
        self.log_var = nn.Linear(encoder_units[2], latent_dim) #Latent logvar

        self.fc4 = nn.Linear(latent_dim, encoder_units[2])
        self.fc5 = nn.Linear(encoder_units[2], encoder_units[1])
        self.fc6 = nn.Linear(encoder_units[1], encoder_units[0])
        self.fc7 = nn.Linear(encoder_units[0], in_dim) #Decoder

    def encode(self, x):
        # print(f'encode x: {x}, {x.shape}')
        x1 = x[:, :noX] # selects the first noX columns
        x2 = x[:, noX:] # selects the remaining columns
        
        
        print(f'encode x1: {x1}, {x1.shape}')
        print(f'encode x2: {x2}, {x2.shape}')    
        
        # estimate a mu_x1 and log_var_x1 (using GMM), but pass x1 thru at the same time
        gmm_x1 = GaussianMixture(n_components=x1.shape[1], covariance_type='diag') # !!! doublecheck whether n_classes is correct here
        gmm_x1.fit(x1.clone().cpu().detach().numpy())
        mu_x1 = torch.from_numpy(gmm_x1.means_).float().to(device)
        log_var_x1 = torch.log(torch.from_numpy(gmm_x1.covariances_)).float().to(device)        
        print(f'encode: mu_x1  {mu_x1}, {mu_x1.shape}')    
        print(f'encode: log_var_x1  {log_var_x1}, {log_var_x1.shape}')    
        
        h = F.relu(self.fc1(x2))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        
        mu_x2 = self.mu(h)
        log_var_x2 = self.log_var(h)
        print(f'encode: mu_x2  {mu_x2}, {mu_x2.shape}')
        print(f'encode: log_var_x2  {log_var_x2}, {log_var_x2.shape}')

        
        mu = torch.cat((mu_x1, mu_x2), axis = 1)
        log_var = torch.cat((log_var_x1, log_var_x2), axis = 1)
        return mu, log_var, x1 

    def decode(self, z):
        z1 = z[:, :noX] # selects the first noX columns
        z2 = z[:, noX:] # selects the remaining columns
        h = F.relu(self.fc4(z2))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        return torch.cat((z1, self.fc7(h)), dim = 1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var, z1 = self.encode(x)
        z2 = self.reparameterize(mu[:, noX:], log_var[:, noX:])
        z = torch.cat((z1, z2), dim=1)
        x_hat = self.decode(z)
        return x_hat, mu, log_var, z


class Autoencoder(torch.nn.Module):
    def __init__(self, in_dim=in_dim, latent_dim=latent_dim):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, encoder_units[0]) #Encoder
        self.fc2 = nn.Linear(encoder_units[0], encoder_units[1])
        self.fc3 = nn.Linear(encoder_units[1], encoder_units[2]) 

        self.mu = nn.Linear(encoder_units[2], latent_dim) #Latent code

        self.fc4 = nn.Linear(latent_dim, encoder_units[2]) 
        self.fc5 = nn.Linear(encoder_units[2], encoder_units[1])
        self.fc6 = nn.Linear(encoder_units[1], encoder_units[0])
        self.fc7 = nn.Linear(encoder_units[0], in_dim) #Decoder

    # def encode(self, x):
    #     h = F.relu(self.fc1(x))
    #     h = F.relu(self.fc2(h))
    #     h = F.relu(self.fc3(h))
    #     return self.mu(h)

    # def decode(self, z):
    #     h = F.relu(self.fc4(z))
    #     h = F.relu(self.fc5(h))
    #     h = F.relu(self.fc6(h))
    #     return F.sigmoid(self.fc7(h))

    def encode(self, x):
        # print(f'encode x: {x}, {x.shape}')
        x1 = x[:, :noX] # selects the first noX columns
        x2 = x[:, noX:] # selects the remaining columns
        print(f'encode x1: {x1}, {x1.shape}')
        print(f'encode x2: {x2}, {x2.shape}')       
        h = F.relu(self.fc1(x2))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return torch.cat((x1, self.mu(h)), axis = 1)

    def decode(self, z):
        z1 = z[:, :noX] # selects the first noX columns
        z2 = z[:, noX:] # selects the remaining columns
        h = F.relu(self.fc4(z2))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        return torch.cat((z1, self.fc7(h)), dim = 1)


    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat