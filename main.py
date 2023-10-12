# for _ in range(10):

root_folder = 'C:/Users/pfbur/Box/projects/CFL-GIP/'
import os
os.chdir(root_folder + 'VaDE_code/Pytorch-VaDE')

results_folder = root_folder + 'results/'
plot_folder = results_folder + 'plots/'

import argparse 
import torch.utils.data
from torchvision import datasets, transforms

from train import TrainerVaDE
from preprocess import get_mnist
from get_toy_data import get_toy_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
time_str = time.strftime("%Y%m%d-%H%M%S")

from global_settings import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of iterations")
    parser.add_argument("--epochs_autoencoder", type=int, default=50,
                        help="number of epochs autoencoder")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--batch_size", type=int, default=500, 
                        help="Batch size")
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='learning rate')
    parser.add_argument('--pretrained_path', type=str, default='weights/pretrained_parameter.pth',
                        help='Output path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # dataloader = get_mnist(batch_size=args.batch_size)
    dataloader = get_toy_data(batch_size=args.batch_size)
    
    vade = TrainerVaDE(args, device, dataloader)
    if args.pretrain==True:
        vade.pretrain()
    vade.train()



# def plot_losses(self):
#     plt.plot(np.asarray(vade.losses))
#     plt.title('Loss per epoch')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.show()
    
# plot_losses(vade)

training_stats = vade.training_stats

loss_direction = dict({'total': 'min',
     'mse_x': 'min', 
     'mse_y': 'min', 
     'log_p_z_given_c': 'min', 
     'log_p_c': 'max', 
     'log_q_c_given_x': 'min', 
     'log_q_z_given_x': 'min', 
     'acc': 'min'})

def plot_losses(self):
    num_plots = len(self.losses)
    num_cols = 2  # You can adjust this to change the number of columns in the grid
    num_rows = num_plots // num_cols + (num_plots % num_cols > 0)
    
    gamma_pred = self.training_stats["gamma_pred"]
    p_c = self.training_stats["p_c"]

    fig, axs = plt.subplots(num_rows + 1, num_cols, figsize=(10, 9))
    axs = axs.flatten()  # To handle the case where num_rows or num_cols is 1
    
    # Count the occurrences of 0s, 1s, 2s, and 3s in each tensor for gamma_pred
    counts_gamma_pred = [torch.bincount(x, minlength=4) for x in gamma_pred]

    # Convert the counts to shares for gamma_pred
    shares_gamma_pred = [count.float() / count.sum() for count in counts_gamma_pred]

    for i, (loss_name, loss_values) in enumerate(self.losses.items()):
        if loss_direction[loss_name] == 'max':
            loss_values = np.asarray(loss_values) * (-1)
        axs[i].plot(loss_values, label=loss_name)
        axs[i].set_title(loss_name)
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Loss')
        
        
    axs[8].set_title('p(c|z) = q(c|x)')
    for i in range(4):
        axs[8].plot([x[i].item() for x in shares_gamma_pred], label=f'Share of {i}s')
    axs[8].set_xlabel('Components of gamma_pred')
    axs[8].set_ylabel('Share')
    # axs[8].legend()
    
    
    # Create the line plot for p_c
    axs[9].set_title('prior pi_c')
    for i in range(4):
        axs[9].plot([x[i].item() for x in p_c], label=f'Share of {i}s')
    axs[9].set_xlabel('Components of p_c')
    axs[9].set_ylabel('Share')
    # axs[9].legend()

    plt.tight_layout()
    plt.show()
    fig.savefig(plot_folder + "model_" + time_str  + "_losses.pdf", 
                bbox_inches='tight',
                dpi = 333)   # save the figure to file     
    
plot_losses(vade)


import matplotlib.pyplot as plt
import numpy as np




loss_dict = vade.losses

def min_max_dict(d):
    result = {}
    for key, values in d.items():
        if key == "acc":
            break
        result[key] = {'min': min(values), 'max': max(values)}
    return result


min_max = pd.DataFrame(min_max_dict(loss_dict)).T
min_max.columns = ['Min', 'Max']
print(min_max)

for item in loss_dict.items():
    print(len(item[1]))
loss_df = pd.DataFrame(loss_dict)

acc_list = loss_dict['acc']

runcell(1, 'C:/Users/pfbur/Box/projects/CFL-GIP/VaDE_code/Pytorch-VaDE/analysis.py')