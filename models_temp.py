import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from global_settings import *

class VaDE(torch.nn.Module):
    def __init__(self, in_dim=in_dim, latent_dim=10, n_classes=10, encoder_units = encoder_units, decoder_units = decoder_units):
        super(VaDE, self).__init__()
        # hidden_units = [512,512,2048,latent_dim,2048,512,512,512]
        self.pi_prior = Parameter(torch.ones(n_classes)/n_classes)
        self.mu_prior = Parameter(torch.zeros(n_classes, latent_dim))
        self.log_var_prior = Parameter(torch.randn(n_classes, latent_dim))
    
 
        # Encoder
        self.encoder = nn.Sequential()
        for i in range(len(encoder_units)):
            # print(hidden_units[i-1], hidden_units[i])
            self.encoder.add_module(f'fc{i+1}', nn.Linear(in_dim if i == 0 else encoder_units[i-1], encoder_units[i]))
            self.encoder.add_module(f'relu{i+1}', nn.ReLU())


        # Latent mu and logvar
        self.mu = nn.Linear(encoder_units[-1], latent_dim)
        self.log_var = nn.Linear(encoder_units[-1], latent_dim)

        # Decoder
        self.decoder = nn.Sequential()
        for i in range(len(decoder_units)):
            # print(decoder_units[i-1], decoder_units[i])
            self.decoder.add_module(f'fc{i+1}', nn.Linear(latent_dim if i == 0 else decoder_units[i-1], decoder_units[i]))
            self.decoder.add_module(f'relu{i+1}', nn.ReLU())
        self.decoder.add_module(f'fc{len(decoder_units)+1}', nn.Linear(decoder_units[-1], in_dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

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
    def __init__(self, in_dim=in_dim, latent_dim=10, encoder_units = encoder_units, decoder_units = decoder_units):
        super(Autoencoder, self).__init__()

 
        # Encoder
        self.encoder = nn.Sequential()
        for i in range(len(encoder_units)):
            # print(hidden_units[i-1], hidden_units[i])
            self.encoder.add_module(f'fc{i+1}', nn.Linear(in_dim if i == 0 else encoder_units[i-1], encoder_units[i]))
            self.encoder.add_module(f'relu{i+1}', nn.ReLU())


        # Latent mu and logvar
        self.mu = nn.Linear(encoder_units[-1], latent_dim)

        # Decoder
        self.decoder = nn.Sequential()
        for i in range(len(decoder_units)):
            # print(decoder_units[i-1], decoder_units[i])
            self.decoder.add_module(f'fc{i+1}', nn.Linear(latent_dim if i == 0 else decoder_units[i-1], decoder_units[i]))
            self.decoder.add_module(f'relu{i+1}', nn.ReLU())
        self.decoder.add_module(f'fc{len(decoder_units)+1}', nn.Linear(decoder_units[-1], in_dim))


    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h)

    def decode(self, z):
        h = self.decoder(z)
        return F.sigmoid(h)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat