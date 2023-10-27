import math
import torch
import numpy as np
from torch import optim
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
# from sklearn.utils.linear_assignment_ import linear_assignment # https://stackoverflow.com/questions/62390517/no-module-named-sklearn-utils-linear-assignment
from scipy.optimize import linear_sum_assignment# as linear_assignment
from models import Autoencoder, VaDE

from global_settings import *

import pdb


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
class TrainerVaDE:
    """This is the trainer for the Variational Deep Embedding (VaDE).
    """
    def __init__(self, args, device, train_data, test_data):
        self.autoencoder = Autoencoder().to(device)
        self.VaDE = VaDE().to(device)
        self.dataloader = train_data
        self.test_dataloader = test_data
        self.device = device
        self.args = args
        self.losses = {'total': [], 
                       'mse_x': [], 
                       'mse_y': [], 
                       'log_p_z_given_c': [], 
                       'log_p_c': [], 
                       'log_q_c_given_x': [], 
                       'log_q_z_given_x': [], 
                       'acc': []}
        self.training_stats = {'p_c': [],
                               'gamma_pred': []}
        self.losses_test = {'total': [], 
                       'mse_x': [], 
                       'mse_y': [], 
                       'log_p_z_given_c': [], 
                       'log_p_c': [], 
                       'log_q_c_given_x': [], 
                       'log_q_z_given_x': [], 
                       'acc': []}
        self.training_stats_test = {'p_c': [],
                               'gamma_pred': []}
        self.vae_loss = {'loss': []}
        self.vae_loss_test = {'loss': []}
        
        

    def pretrain(self):
        """Here we train an stacked autoencoder which will be used as the initialization for the VaDE. 
        This initialization is usefull because reconstruction in VAEs would be weak at the begining
        and the models are likely to get stuck in local minima.
        """
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.002)
        self.autoencoder.apply(weights_init_normal) #intializing weights using normal distribution.
        self.autoencoder.train()
        print('Training the autoencoder...')
        for epoch in range(self.args.epochs_autoencoder): # used to be range(30)
            total_loss = 0
            for x, _ in self.dataloader:
                optimizer.zero_grad()
                x = x.to(self.device)
                # print(f'autoenc: shape of x: {x.shape}')
                x_hat = self.autoencoder(x)
                # loss = F.binary_cross_entropy(x_hat, x, reduction='mean') # just reconstruction
                loss = F.mse_loss(x_hat, x, reduction='mean')
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.vae_loss['loss'].append(total_loss/len(self.dataloader))
            self.autoencoder.eval()
            with torch.no_grad():
                total_loss = 0
                for x, _ in self.test_dataloader:
                    x = x.to(self.device)
                    x_hat = self.autoencoder(x)
                    # loss = F.binary_cross_entropy(x_hat, x, reduction='mean') # just reconstruction
                    loss = F.mse_loss(x_hat, x, reduction='mean')
                    total_loss += loss.item()
                self.vae_loss_test['loss'].append(total_loss/len(self.test_dataloader))
        self.train_GMM() #training a GMM for initialize the VaDE
        self.save_weights_for_VaDE() #saving weights for the VaDE


    def train_GMM(self):
        """It is possible to fit a Gaussian Mixture Model (GMM) using the latent space 
        generated by the stacked autoencoder. This way, we generate an initialization for 
        the priors (pi, mu, var) of the VaDE model.
        """
        print('Fiting Gaussian Mixture Model...')
        # x = torch.cat([data[0] for data in self.dataloader]).view(-1, in_dim).to(self.device) #all x samples.
        
        # this is the updated version, we now just take the whole data with all dimensions as input for the GMM
        x = torch.cat([data[0] for data in self.dataloader]).to(self.device) #all x samples.

        print(f'GMM: shape of x {x.shape}')
        z = self.autoencoder.encode(x)
        print(f'GMM: shape of z {z.shape}')
        self.gmm = GaussianMixture(n_components=n_classes, covariance_type='diag') # !!! doublecheck whether n_classes is correct here
        self.gmm.fit(z.cpu().detach().numpy())
        
        # here, estimate another GMM


    def save_weights_for_VaDE(self):
        """Saving the pretrained weights for the encoder, decoder, pi, mu, var.
        """
        print('Saving weights.')
        state_dict = self.autoencoder.state_dict()

        self.VaDE.load_state_dict(state_dict, strict=False)
        self.VaDE.pi_prior.data = torch.log(torch.from_numpy(self.gmm.weights_).float().to(self.device))
        self.VaDE.mu_prior.data = torch.from_numpy(self.gmm.means_).float().to(self.device)
        self.VaDE.log_var_prior.data = torch.log(torch.from_numpy(self.gmm.covariances_)).float().to(self.device)
        torch.save(self.VaDE.state_dict(), self.args.pretrained_path)    

    def train(self):
        """
        """
        if self.args.pretrain==True:
            self.VaDE.load_state_dict(torch.load(self.args.pretrained_path,
                                                 map_location=self.device))
        else:
            self.VaDE.apply(weights_init_normal)
        self.optimizer = optim.Adam(self.VaDE.parameters(), lr=self.args.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=10, gamma=0.9)
        print('Training VaDE...')
        for epoch in range(self.args.epochs):
            self.train_VaDE(epoch)
            self.test_VaDE(epoch)
            lr_scheduler.step()


    def train_VaDE(self, epoch):
        self.VaDE.train()

        total_loss = 0
        for x, true_label in self.dataloader:
            self.optimizer.zero_grad()
            x = x.to(self.device)
            x_hat, mu, log_var, z = self.VaDE(x)
            #print('Before backward: {}'.format(self.VaDE.pi_prior))
            
            # here we need to estimate mu_x1 and sigma_x1
            # design choice: do we do this here, in the training loop,
            # or do we do that on all the data, once, in the pretraining
            # >> decision: do it once in the pretraining
            
            loss, loss_components, training_stats = self.compute_loss(x, x_hat, mu, log_var, z, epoch, true_label)
            # print(loss)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
            for loss_name, loss_value in loss_components.items():
                self.losses[loss_name].append(loss_value)
            for training_stats_name, training_stats_value in training_stats.items():
                self.training_stats[training_stats_name].append(training_stats_value)

            #print('After backward: {}'.format(self.VaDE.pi_prior))
        print('Training VaDE... Epoch: {}, Loss: {}'.format(epoch, total_loss))


    def test_VaDE(self, epoch):
        self.VaDE.eval()
        with torch.no_grad():
            total_loss = 0
            y_true, y_pred = [], []
            for x, true_label in self.test_dataloader:
                x = x.to(self.device)
                x_hat, mu, log_var, z = self.VaDE(x)
                # compute loss
                if verbatim == 1:
                    print(f'testvade: shapes  of z, pi_prior: {z.shape}, {self.VaDE.pi_prior.shape}')
                gamma = self.compute_gamma(z, torch.exp(self.VaDE.pi_prior))
                pred = torch.argmax(gamma, dim=1)
                loss, loss_components, training_stats = self.compute_loss(x, x_hat, mu, log_var, z, epoch, true_label)
                total_loss += loss.item()
                y_true.extend(true_label.numpy())
                y_pred.extend(pred.cpu().detach().numpy())
                for loss_name, loss_value in loss_components.items():
                    # print(f'loss name in test vade is {loss_name}, {loss_value}')
                    self.losses_test[loss_name].append(loss_value)
                    # print(self.losses_test[loss_name])
                for training_stats_name, training_stats_value in training_stats.items():
                    self.training_stats_test[training_stats_name].append(training_stats_value)
            acc = self.cluster_acc(np.array(y_true), np.array(y_pred))
            # add accuracy to the loss components
            # print(acc)
            # self.losses["acc"].append(acc.item())
            
            print('Testing VaDE... Epoch: {}, Loss: {}, Acc: {}'.format(epoch, total_loss, acc))


    def compute_loss(self, xy, xy_hat, mu, log_var, z, epoch, true_label):
        no_epochs = self.args.epochs
        weight = self.compute_weight(no_epochs, epoch)
        weight2 = self.compute_weight2(no_epochs, epoch)
        weights = self.compute_weights(no_epochs, epoch, no_weights = 9)
        x = xy[:, :dim_x] # selects the first dim_x columns
        y = xy[:, dim_x:] # selects the remaining columns
        x_hat = xy_hat[:, :dim_x] # selects the first dim_x columns
        y_hat = xy_hat[:, dim_x:] # selects the remaining columns
        # p_c = torch.sigmoid(self.VaDE.pi_prior)
        # p_c = self.VaDE.pi_prior # this sometimes has negative values, so we need a fix
        # p_c = torch.clamp(self.VaDE.pi_prior, min=1e-9) # just clamp it !!! double check whether there is a better solution
        # p_c = torch.sigmoid(self.VaDE.pi_prior)
        p_c = torch.exp(self.VaDE.pi_prior) 
        gamma = self.compute_gamma(z, p_c) # nobs x no_classes, gamma is q_c_given_x = p_c_given_z
        # if verbatim == 1:
            # print(f'min,max of z {z.min(), z.max()}')
            # print(f'shape of gamma in l {gamma.shape}')
            # print(f'min max of gamma is {gamma.min(), gamma.max()}')
            # print(f'min, max of x     is {round(x.min().item(), 4)}, {round(x.max().item(), 4)}')
            # print(f'min, max of x_hat is {round(x_hat.min().item(), 4)}, {round(x_hat.max().item(), 4)}')
            # print(f'min,max of p_c {p_c.min(), p_c.max()}')
            # print(f'compute l: vade pi prior is {p_c}')
            # print(f'compute l: shape of log_var: {log_var.shape}')
            # print(f'compute l: shape of mu: {mu.shape}')
            # print(f'shape of p_c in l {p_c.shape}')
            # print(f'epoch is {epoch} of {no_epochs}, weight is {weight}')
            
        # log_p_x_given_z = F.binary_cross_entropy(x_hat, x, reduction='sum') 
        # mse_x = F.binary_cross_entropy(x_hat, x, reduction='sum') 
        # mse_y = F.binary_cross_entropy(y_hat, y, reduction='sum') 
        mse_x = F.mse_loss(x_hat, x, reduction='sum')
        mse_y = F.mse_loss(y_hat, y, reduction='sum') # used to be called log_p_x_given_z 
        h = log_var.exp().unsqueeze(1) + (mu.unsqueeze(1) - self.VaDE.mu_prior).pow(2) # mu here needs the same dimensionality as VaDE.mu_prior
        h = torch.sum(self.VaDE.log_var_prior + h / self.VaDE.log_var_prior.exp(), dim=2) # obs x n_classes x latent_dim
        log_p_z_given_c = 0.5 * torch.sum(gamma * h) # ok, see eq. B --- SCALAR
        log_p_c = torch.sum(gamma * torch.log(p_c/p_c.sum() + 1e-9)) # ok, see eq. C in Appendix -- added the division by p_c sum here because p_c is not constrained to add up to 1 -- this is not important for compute_gamma because compute_gamma(z, p_c) = compute_gamma(z, p_c * constant)
        log_q_c_given_x = torch.sum(gamma * torch.log(gamma + 1e-9)) # eq. E in Appendix
        log_q_z_given_x = -0.5 * torch.sum(1 + log_var) # ok, see eq. D in App., added a minus sign here and changed the sign for this component in the overall loss (original code is fine, this is just for better readability)

        self.args.weight_regulariser
        loss = 10000 * mse_x + 10000 * weights[1] * mse_y + self.args.weight_regulariser * weights[2] * log_p_z_given_c - self.args.weight_regulariser * weights[4] * log_p_c + self.args.weight_regulariser * weights[4] * log_q_c_given_x + self.args.weight_regulariser * weights[4] * log_q_z_given_x # changed the signs, 
        # loss = mse_x + weights[1] * mse_y #- weight* log_p_c #+ weight * log_p_z_given_c #+ log_q_c_given_x + log_q_z_given_x # changed the signs, 
        # loss = loss - weights[2] * log_p_c + weights[3] * log_p_z_given_c + weights[6] * log_q_c_given_x 
        # loss = mse_x +  weights[1] * mse_y +  weights[2] * log_p_z_given_c -  weights[3] * log_p_c +  weights[4] * log_q_c_given_x +  weights[5] *  log_q_z_given_x
        #old:  log_p_x_given_z + log_p_z_given_c - log_p_c + log_q_c_given_x - log_q_z_given_x
        # 
        loss /= x.size(0)
        
        # compute accuracy    
        acc = self.cluster_acc(np.array(true_label.numpy()), np.array(torch.argmax(gamma, dim=1).cpu().detach().numpy()))
        # Assuming 'loss' is your loss variable
        if np.isnan(loss.detach().numpy()):
            pdb.set_trace()
        
        loss_components = {'total': loss.item(), 
                           'mse_x': mse_x.item(), 
                           'mse_y': mse_y.item(), 
                           'log_p_z_given_c': log_p_z_given_c.item(), 
                           'log_p_c': log_p_c.item(), 
                           'log_q_c_given_x': log_q_c_given_x.item(), 
                           'log_q_z_given_x': log_q_z_given_x.item(),
                           'acc': acc}
        training_stats = {'p_c' : p_c,
                          'gamma_pred' : torch.argmax(gamma, dim=1)}
        return loss, loss_components, training_stats
    


    def compute_weight(self, no_epochs, epoch):
        if epoch < no_epochs * 2 / 8:
            weight = 0
        elif epoch < no_epochs * 4 / 8:
            weight = (epoch - no_epochs * 2 / 8) / (no_epochs * 2 / 8)
        else:
            weight = 1
        return weight
        # if epoch < no_epochs / 4:
        #     weight = 0
        # elif epoch < no_epochs * 3 / 4:
        #     weight = 2 * (epoch - no_epochs / 4) / no_epochs
        # else:
        #     weight = 1
        # return weight
    def compute_weight2(self, epoch, no_epochs):
        if epoch < no_epochs * 4 / 8:
            weight = 0
        elif epoch < no_epochs * 6 / 8:
            weight = (epoch - no_epochs * 4 / 8) / (no_epochs * 2 / 8)
        else:
            weight = 1
        return weight

    def compute_weights(self, epoch, no_epochs, no_weights):
        weights = [0] * no_weights
        interval_length = (no_epochs * 3 / 4) / no_weights

        for i in range(no_weights):
            interval_start = i * interval_length
            interval_end = (i + 1) * interval_length

            if epoch < interval_start:
                weights[i] = 0
            elif epoch < interval_end:
                weights[i] = (epoch - interval_start) / interval_length
            else:
                weights[i] = 1

        return weights


    def compute_gamma(self, z, p_c):
        # print(f'compute gamma: shape of z {z.shape}')
        # print(f'compute gamma: shape of p_c {p_c.shape}')
        # print(f'compute gamma: min, max of p_c {p_c.min(), p_c.max()}')
        h = (z.unsqueeze(1) - self.VaDE.mu_prior).pow(2) / self.VaDE.log_var_prior.exp()
        # !!! are those VaDE priors updated???
        h += self.VaDE.log_var_prior
        h += torch.Tensor([np.log(np.pi*2)]).to(self.device)
        # print(f'compute gamma: shape of h {h.shape}') # h has shape (no_obs_batch, no_classes, latent_dim)
        p_z_c = torch.exp(torch.log(p_c + 1e-9).unsqueeze(0) - 0.5 * torch.sum(h, dim=2)) + 1e-9 # sum over latent dim
        # p_z_c size is nobs x n classes
        gamma = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True) # this is equation 16 in the paper and the standard proba of a gaussian
        # print(f'compute gamma: shape of gamma {gamma.shape}')
        return gamma

    def cluster_acc(self, real, pred):
        D = max(pred.max(), real.max())+1
        w = np.zeros((D,D), dtype=np.int64)
        for i in range(pred.size):
            # print(f'cluster acc: pred[i] is {pred[i]}, real[i] is {real[i]}')
            w[pred[i], real[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        total_cost = w[ind[0], ind[1]].sum()
        return total_cost * 1.0 / pred.size * 100#, w # chat gpt correction!


"""
    def compute_loss(self, x, x_hat, mu, log_var, z):
        p_c = self.VaDE.pi_prior
        gamma = self.compute_gamma(z, p_c)

        log_p_x_given_z = F.binary_cross_entropy(x_hat, x, reduction='sum') * x.size(1)
        log_p_z_given_c = 0.5 * torch.sum(gamma * torch.sum(mu.size(-1)*np.log(2*np.pi)\
                          + self.VaDE.log_var_prior.unsqueeze(0)\
                          + log_var.exp().unsqueeze(1)/self.VaDE.log_var_prior.exp()\
                          + (mu.unsqueeze(1) - self.VaDE.mu_prior).pow(2)/self.VaDE.log_var_prior.exp(), dim=2))
        log_p_c = torch.sum(gamma * torch.log(p_c + 1e-9))
        log_q_c_given_x = torch.sum(gamma * torch.log(gamma + 1e-9))
        log_q_z_given_x = 0.5 * torch.sum(1 + log_var)

        loss = log_p_x_given_z + log_p_z_given_c - log_p_c + log_q_c_given_x - log_q_z_given_x
        loss /= x.size(0)

        return loss

    def compute_gamma(self, z, p_c):
        h = (z.unsqueeze(1) - self.VaDE.mu_prior).pow(2) / self.VaDE.log_var_prior.exp() + \
             np.log(2*np.pi) + self.VaDE.log_var_prior
        p_z_c = torch.exp(torch.log(p_c + 1e-9).unsqueeze(0) - 0.5 * torch.sum(h, dim=2))
        gamma = p_z_c / torch.sum(p_z_c + 1e-9, dim=1, keepdim=True)
        return gamma
"""
