

import os
os.chdir('C:/Users/pfbur/Box/projects/CFL-GIP/VaDE_code/Pytorch-VaDE')

import argparse 
import torch.utils.data
from torchvision import datasets, transforms

from train import TrainerVaDE
from preprocess import get_mnist
from get_toy_data import get_toy_data


from global_settings import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30,
                        help="number of iterations")
    parser.add_argument("--epochs_autoencoder", type=int, default=5,
                        help="number of epochs autoencoder")
    parser.add_argument("--patience", type=int, default=50, 
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='learning rate')
    parser.add_argument("--batch_size", type=int, default=200, 
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



#%%
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


from hsic_torch import *
# how to get the model params out?


df = pd.read_csv('toy_data.csv')

df.head()


df1 = df[["X1", "X2", "Y"]]
df1 = torch.from_numpy(df1.values)
df1 = df1.float()


# feed through model
xy_hat, mu, log_var, z = vade.VaDE(df1)


# disentangle
df1 = df1.numpy()
xy_hat = xy_hat.detach().numpy()
z = z.detach().numpy()

xhat = xy_hat[:,:dim_x]
yhat = xy_hat[:,dim_x:]

z1 = z[:,:latent_dim_x]
z2 = z[:,latent_dim_x:]


# are Z1 and Z2 dependent?

## this HSIC implementation is not working, test statistic scales with number of observations!!

# nest step, try this: https://github.com/Black-Swan-ICL/PyRKHSstats/tree/main

hsic_gam_torch(torch.from_numpy(z1), torch.from_numpy(z2))

hsic_gam_torch(torch.from_numpy(z1[1:100]), torch.from_numpy(z2[1:100]))

hsic_gam_torch(torch.from_numpy(z1[1:10]), torch.from_numpy(z2[1:10]))


testStat, thresh = hsic_gam(torch.from_numpy(z1), torch.from_numpy(z2), alph = 0.05)
## this is computing a test stat but it doesnt produce a p value....

df_hat = pd.concat((pd.DataFrame(df1), pd.DataFrame(z)), axis = 1)

col_names = ["X"+str(i+1) for i in range(dim_x)] + ["Y"] + ["ZX"+str(i+1) for i in range(latent_dim_x)] + ["ZY"+str(i+1) for i in range(latent_dim_y)]


df_hat.columns = col_names


# how good is the reconstruction?
sns.pairplot(pd.concat((pd.DataFrame(df1), pd.DataFrame(df_hat[["X1", "X2", "Y"]])), axis = 1))


# does the latent learn anything?
sns.pairplot(df_hat)



# naive regression

# Get all columns that start with 'X'
x_cols = [col for col in df.columns if col.startswith('X')]

# Add a constant to the independent values
X = sm.add_constant(df[x_cols])

# Fit the model
model1 = sm.OLS(df['Y'], X).fit()

# Print out the statistics
print(model1.summary())


## true model (with known confounder)
# Get all columns that start with 'X' and 'L'
x_l_cols = x_cols + [col for col in df.columns if col.startswith('L')]

# Add a constant to the independent values
X = sm.add_constant(df[x_l_cols])

# Fit the model
model2 = sm.OLS(df['Y'], X).fit()

# Print out the statistics
print(model2.summary())


## now estimate the model with recoverd confounder
# Get all columns that start with 'X' and 'L'
x_z_cols = x_cols + [col for col in df_hat.columns if col.startswith('ZY')]

# Add a constant to the independent values
X = sm.add_constant(df_hat[x_z_cols])

# Fit the model
model3 = sm.OLS(df_hat['Y'], X).fit()

# Print out the statistics
print(model3.summary())





