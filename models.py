import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from global_settings import *



class VaDE(torch.nn.Module):
    def __init__(self, in_dim=in_dim, latent_dim=5, n_classes=10): # latent_dim used to be 10
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
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.mu(h), self.log_var(h)

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        return F.sigmoid(self.fc7(h))

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
    def __init__(self, in_dim=in_dim, latent_dim=10):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, encoder_units[0]) #Encoder
        self.fc2 = nn.Linear(encoder_units[0], encoder_units[1])
        self.fc3 = nn.Linear(encoder_units[1], encoder_units[2]) 

        self.mu = nn.Linear(encoder_units[2], latent_dim) #Latent code

        self.fc4 = nn.Linear(latent_dim, encoder_units[2]) 
        self.fc5 = nn.Linear(encoder_units[2], encoder_units[1])
        self.fc6 = nn.Linear(encoder_units[1], encoder_units[0])
        self.fc7 = nn.Linear(encoder_units[0], in_dim) #Decoder

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.mu(h)

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        return F.sigmoid(self.fc7(h))

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat