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


import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# load real data

df = pd.read_csv(root_folder + "JS2018\\data\\taste_of_wine\\winequality-red.csv", sep = ';')

# Sample DataFrame, you've already loaded yours as df
# df = pd.read_csv('path_to_your_file.csv', delimiter=';')

# Define independent variables (all columns except 'quality')
X = df.drop('quality', axis=1)

# Add a constant (i.e., bias or intercept)
X = sm.add_constant(X)

# Define dependent variable
y = df['quality']

# Run regression
model = sm.OLS(y, X).fit()

# Print out the statistics
print(model.summary())


# Define independent variables (all columns except 'quality')
X = df.drop('quality', axis=1)

# Standardize the independent variables
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Add a constant (i.e., bias or intercept) to the standardized data
X_standardized = sm.add_constant(X_standardized)

# Define dependent variable
y = df['quality']

# Run regression
model = sm.OLS(y, X_standardized).fit()

# Print out the statistics
print(model.summary())
