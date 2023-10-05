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
    parser.add_argument("--epochs", type=int, default=150,
                        help="number of iterations")
    parser.add_argument("--epochs_autoencoder", type=int, default=1,
                        help="number of epochs autoencoder")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--batch_size", type=int, default=500, 
                        help="Batch size")
    parser.add_argument('--pretrain', type=bool, default=False,
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

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 6))
    axs = axs.flatten()  # To handle the case where num_rows or num_cols is 1

    for i, (loss_name, loss_values) in enumerate(self.losses.items()):
        if loss_direction[loss_name] == 'max':
            loss_values = np.asarray(loss_values) * (-1)
        axs[i].plot(loss_values, label=loss_name)
        axs[i].set_title(loss_name)
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Loss')

    plt.tight_layout()
    plt.show()
    fig.savefig(plot_folder + "model_" + time_str  + "_losses.pdf", 
                bbox_inches='tight',
                dpi = 333)   # save the figure to file     
    
plot_losses(vade)



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

runcell(1, 'C:/Users/pfbur/Box/projects/CFL-GIP/VaDE_code/Pytorch-VaDE/analysis.py')