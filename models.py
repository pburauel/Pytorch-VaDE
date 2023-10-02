import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from sklearn.mixture import GaussianMixture

from global_settings import *



class VaDE(torch.nn.Module):
    def __init__(self, dim_x = dim_x, dim_y = dim_y, latent_dim_x = latent_dim_x, latent_dim_y = latent_dim_y, n_classes=n_classes): # latent_dim used to be 10
        super(VaDE, self).__init__()

        self.pi_prior = Parameter(torch.ones(n_classes)/n_classes)
        self.mu_prior = Parameter(torch.zeros(n_classes, latent_dim_x + latent_dim_y))
        self.log_var_prior = Parameter(torch.randn(n_classes, latent_dim_x + latent_dim_y))
        
        self.fc1x = nn.Linear(dim_x, encoder_units[0]) #Encoder
        self.fc2x = nn.Linear(encoder_units[0], encoder_units[1])
        self.fc3x = nn.Linear(encoder_units[1], encoder_units[2]) 

        self.mu_x = nn.Linear(encoder_units[2], latent_dim_x) #Latent mu
        self.log_var_x = nn.Linear(encoder_units[2], latent_dim_x) #Latent logvar

        self.fc4x = nn.Linear(latent_dim_x, encoder_units[2])
        self.fc5x = nn.Linear(encoder_units[2], encoder_units[1])
        self.fc6x = nn.Linear(encoder_units[1], encoder_units[0])
        self.fc7x = nn.Linear(encoder_units[0], dim_x) #Decoder


        self.fc1y = nn.Linear(dim_y, encoder_units[0]) #Encoder
        self.fc2y = nn.Linear(encoder_units[0], encoder_units[1])
        self.fc3y = nn.Linear(encoder_units[1], encoder_units[2]) 

        self.mu_y = nn.Linear(encoder_units[2], latent_dim_y) #Latent mu
        self.log_var_y = nn.Linear(encoder_units[2], latent_dim_y) #Latent logvar

        self.fc4y = nn.Linear(latent_dim_y, encoder_units[2])
        self.fc5y = nn.Linear(encoder_units[2], encoder_units[1])
        self.fc6y = nn.Linear(encoder_units[1], encoder_units[0])
        self.fc7y = nn.Linear(encoder_units[0], dim_y) #Decoder
    def encode(self, xy):
        # print(f'encode x: {x}, {x.shape}')
        x = xy[:, :dim_x] # selects the first dim_x columns
        y = xy[:, dim_x:] # selects the remaining columns
        
        if verbatim == 1:
            print(f'encode x: {x}, {x.shape}')
            print(f'encode y: {y}, {y.shape}')    
        
        # estimate a mu_x1 and log_var_x1 (using GMM), but pass x1 thru at the same time
        # gmm_x1 = GaussianMixture(n_components=x1.shape[1], covariance_type='diag') # !!! doublecheck whether n_classes is correct here
        # gmm_x1.fit(x1.clone().cpu().detach().numpy())
        # mu_x1 = torch.from_numpy(gmm_x1.means_).float().to(device)
        # log_var_x1 = torch.log(torch.from_numpy(gmm_x1.covariances_)).float().to(device)        
        # print(f'encode: mu_x1  {mu_x1}, {mu_x1.shape}')    
        # print(f'encode: log_var_x1  {log_var_x1}, {log_var_x1.shape}')    
        
        hx = F.relu(self.fc1x(x))
        hx = F.relu(self.fc2x(hx))
        hx = F.relu(self.fc3x(hx))

        hy = F.relu(self.fc1y(y))
        hy = F.relu(self.fc2y(hy))
        hy = F.relu(self.fc3y(hy))
        
        mu_x = self.mu_x(hx)
        log_var_x = self.log_var_x(hx)
        if verbatim == 1:        
            print(f'encode: mu_x  {mu_x}, {mu_x.shape}')
            print(f'encode: log_var_x  {log_var_x}, {log_var_x.shape}')

        mu_y = self.mu_y(hy)
        log_var_y = self.log_var_y(hy)
        if verbatim == 1:
            print(f'encode: mu_y  {mu_y}, {mu_y.shape}')
            print(f'encode: log_var_y  {log_var_y}, {log_var_y.shape}')
        
        mu = torch.cat((mu_x, mu_y), axis = 1)
        log_var = torch.cat((log_var_x, log_var_y), axis = 1)
        return mu, log_var

    def decode(self, z):
        z1 = z[:, :dim_x] # selects the first dim_x columns
        z2 = z[:, dim_x:] # selects the remaining columns
        hx = F.relu(self.fc4x(z1))
        hx = F.relu(self.fc5x(hx))
        hx = F.relu(self.fc6x(hx))
        
        hy = F.relu(self.fc4y(z2))
        hy = F.relu(self.fc5y(hy))
        hy = F.relu(self.fc6y(hy))
        return torch.cat((self.fc7x(hx), self.fc7y(hy)), dim = 1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var, z


class Autoencoder(torch.nn.Module):
    def __init__(self, dim_x = dim_x, dim_y = dim_y, latent_dim_x = latent_dim_x, latent_dim_y = latent_dim_y):
        super(Autoencoder, self).__init__()
        self.fc1x = nn.Linear(dim_x, encoder_units[0]) #Encoder
        self.fc2x = nn.Linear(encoder_units[0], encoder_units[1])
        self.fc3x = nn.Linear(encoder_units[1], encoder_units[2]) 

        self.fc1y = nn.Linear(dim_y, encoder_units[0]) #Encoder
        self.fc2y = nn.Linear(encoder_units[0], encoder_units[1])
        self.fc3y = nn.Linear(encoder_units[1], encoder_units[2]) 

        self.mu_x = nn.Linear(encoder_units[2], latent_dim_x) #Latent code
        self.mu_y = nn.Linear(encoder_units[2], latent_dim_y) #Latent code

        self.fc4x = nn.Linear(latent_dim_x, encoder_units[2]) 
        self.fc5x = nn.Linear(encoder_units[2], encoder_units[1])
        self.fc6x = nn.Linear(encoder_units[1], encoder_units[0])
        self.fc7x = nn.Linear(encoder_units[0], dim_x) #Decoder

        self.fc4y = nn.Linear(latent_dim_y, encoder_units[2]) 
        self.fc5y = nn.Linear(encoder_units[2], encoder_units[1])
        self.fc6y = nn.Linear(encoder_units[1], encoder_units[0])
        self.fc7y = nn.Linear(encoder_units[0], dim_y) #Decoder

    def encode(self, xy):
        # print(f'encode x: {x}, {x.shape}')
        x = xy[:, :dim_x] # selects the first dim_x columns
        y = xy[:, dim_x:] # selects the remaining columns
        if verbatim == 1:
            print(f'encode x: {x}, {x.shape}')
            print(f'encode y: {y}, {y.shape}')       
        hx = F.relu(self.fc1x(x))
        hx = F.relu(self.fc2x(hx))
        hx = F.relu(self.fc3x(hx))

        hy = F.relu(self.fc1y(y))
        hy = F.relu(self.fc2y(hy))
        hy = F.relu(self.fc3y(hy))
        return torch.cat((self.mu_x(hx), self.mu_y(hy)), axis = 1)

    def decode(self, z):
        z_x = z[:, :dim_x] # selects the first dim_x columns
        z_y = z[:, dim_x:] # selects the remaining columns
        
        hx = F.relu(self.fc4x(z_x))
        hx = F.relu(self.fc5x(hx))
        hx = F.relu(self.fc6x(hx))
        
        hy = F.relu(self.fc4y(z_y))
        hy = F.relu(self.fc5y(hy))
        hy = F.relu(self.fc6y(hy))
        return torch.cat((self.fc7x(hx), self.fc7y(hy)), dim = 1)


    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat