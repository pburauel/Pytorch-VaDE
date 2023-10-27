for wr in [1, 1, 1, 1, 1, 1]:
# wr = 10
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
    time_str = time.strftime("%Y%m%d-%H%M%S") + "_wr" + str(wr)
    
    from global_settings import *
    
    
    
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", type=int, default=500,
                            help="number of iterations")
        parser.add_argument("--weight_regulariser", type=int, default=wr,
                            help="number of epochs autoencoder")
        parser.add_argument("--epochs_autoencoder", type=int, default=200,
                            help="number of epochs autoencoder")
        parser.add_argument("--patience", type=int, default=10, 
                            help="Patience for Early Stopping")
        parser.add_argument('--lr', type=float, default=1e-4,
                            help='learning rate')
        parser.add_argument("--batch_size", type=int, default=100, # works well with 100
                            help="Batch size")
        parser.add_argument('--pretrain', type=bool, default=True,
                            help='learning rate')
        parser.add_argument('--pretrained_path', type=str, default='weights/pretrained_parameter.pth',
                            help='Output path')
        args = parser.parse_args()
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = 'cpu'
    
        # dataloader = get_mnist(batch_size=args.batch_size)
        train_loader, test_loader = get_toy_data(batch_size=args.batch_size)
        vade = TrainerVaDE(args, device, train_loader, test_loader)
        if args.pretrain==True:
            vade.pretrain()
        vade.train()
    
    
    ### plot autoencoder losses
    
    fig, axs = plt.subplots(figsize=(10, 9))  # Create a figure and a set of subplots
    
    axs.plot(vade.vae_loss['loss'], label='train')  # Assuming 'vade.vae_loss' and 'vade.vae_loss_test' are dictionaries with key 'loss'
    axs.plot(vade.vae_loss_test['loss'], label='test')
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.legend()  # To show labels in the plot
    plt.show()
    
    # def plot_losses(self):
    #     plt.plot(np.asarray(vade.losses))
    #     plt.title('Loss per epoch')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.show()
        
    # plot_losses(vade)
    
    training_stats = vade.training_stats
    len_train = len(training_stats["p_c"])
    training_stats_test = vade.training_stats_test
    len_test = len(training_stats_test["p_c"])
    
    training_stats = vade.training_stats
    len_train = len(training_stats["p_c"])
    losses_test = vade.losses_test
    # losses_test_temp = losses_test
    
    # vade_losses_temp = losses_test
        
    # Calculate the repeat factor, duplicate values for the test dictionary, since there are fewer observations
    repeat_factor = len(vade.losses['total']) // len(vade.losses_test['total'])
    
    
    for loss_name, loss_values in vade.losses_test.items():
        # Convert the list to a numpy array
        loss_values_array = np.array(loss_values)
        
        # Repeat the elements in the array
        loss_values_repeated = np.repeat(loss_values_array, repeat_factor)
        
        # Convert the repeated array back to a list and update the dictionary
        vade.losses_test[loss_name] = loss_values_repeated.tolist()
    
    
    loss_direction = dict({'total': 'min',
         'mse_x': 'min', 
         'mse_y': 'min', 
         'log_p_z_given_c': 'min', 
         'log_p_c': 'max', 
         'log_q_c_given_x': 'min', 
         'log_q_z_given_x': 'min', 
         'acc': 'min'}) # this is just to make the loss plot visually easier to understand: all pieces of the loss should go DOWN now
    
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
    
    
        for i, (loss_name, loss_values) in enumerate(self.losses_test.items()):
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
        fig.savefig(plot_folder + "model" + time_str  + "_losses.pdf", 
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
    
    
    acc_list = loss_dict['acc']
    
    
    runcell(1, 'C:/Users/pfbur/Box/projects/CFL-GIP/VaDE_code/Pytorch-VaDE/analysis.py')
    runcell('pairplots', 'C:/Users/pfbur/Box/projects/CFL-GIP/VaDE_code/Pytorch-VaDE/analysis.py')
    runcell('deconfound', 'C:/Users/pfbur/Box/projects/CFL-GIP/VaDE_code/Pytorch-VaDE/analysis.py')