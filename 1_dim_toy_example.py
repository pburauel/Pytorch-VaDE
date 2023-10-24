import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
import pandas as pd

root_folder = 'C:/Users/pfbur/Box/projects/CFL-GIP/'
import os
os.chdir(root_folder + 'VaDE_code/Pytorch-VaDE')

results_folder = root_folder + 'results/'
plot_folder = results_folder + 'plots/'

# Set the number of samples n
n = 1000


## option 1 to generate H and L
# Define the joint probability matrix M for H and L
alpha = 0.8 # can be between 0 and 1
beta = alpha - 0.01
gamma = 0.1
M = np.array([[0.5 - (alpha/2), 0.5 - (1-alpha)/2],  # Probabilities for L=0
              [0.5 - (1-beta)/2, 0.5 - (beta/2)]]) # Probabilities for L=1

M = np.array([[0.5 - (alpha/2), 0.5 - (1-alpha)/2 - gamma],  # Probabilities for L=0
              [0.5 + (gamma/2) - (1-beta)/2, 0.5  + (gamma/2) - (beta/2)]]) # Probabilities for L=1

M = np.array([[0.05, 0.35],  # Probabilities for L=0
              [0.5, 0.1]]) # Probabilities for L=1


# Check that M is a valid joint probability matrix (i.e., all entries are non-negative and sum to 1)
assert np.all(M >= 0) and np.isclose(np.sum(M), 1), "Invalid joint probability matrix"

# Compute the marginal probabilities of L and H
P_L = np.sum(M, axis=1)
P_H = np.sum(M, axis=0)

# Sample H and L from the joint distribution
HL = np.random.choice([0, 1, 2, 3], size=n, p=M.flatten())

# M flatten is P(H = 0 | L = 0), P(H = 1 | L = 0), P(H = 0 | L = 1), P(H = 1 | L = 1)

L, H = HL // 2, HL % 2
H = H.reshape(-1,1)
L = L.reshape(-1,1)

# # Generate a one-dimensional confounder H
# H = np.random.normal(size=(n, 1))

# H = np.random.choice([-1, 0, 1], size=n, p=[1/3, 1/3, 1/3]).reshape(-1,1)

# H = np.random.choice([-1, 1], size=n, p=[0.5, 0.5])

# Generate cause X, which is influenced by H
X = L + np.random.normal(size=(n, 1)) * 2

X = np.zeros((n,1))
X[H.reshape(-1,) == 0] = np.random.normal(loc=[.5], scale=[0.2], size=(np.sum(H == 0), 1))
X[H.reshape(-1,) == 1] = np.random.normal(loc=[2], scale=[0.2], size=(np.sum(H == 1), 1))

# X = X + abs(X.min())

# Generate outcome Y, which is a function of X and H
UY = np.random.normal(size=(n, 1)) * .1
Y = X**2 + 3*L + UY

# Fit a naive linear regression model of Y on X
X_naive = sm.add_constant(X)
model_naive = sm.OLS(Y, X_naive).fit()

# Fit a linear regression model of Y on X and H
XL = sm.add_constant(np.concatenate([X, L], axis=1))
model_adjusted = sm.OLS(Y, XL).fit()

# Set the regression estimates
beta_naive = model_naive.params
beta_adjusted = model_adjusted.params

# Generate regression lines for the plot
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
x_line_naive = sm.add_constant(x_line)
x_line_adjusted = sm.add_constant(x_line)
y_line_naive = np.dot(x_line_naive, beta_naive)
y_line_adjusted = np.dot(x_line_adjusted, beta_adjusted[0:2])

# Create three confounder-groups by estimating k-means on H
kmeans = KMeans(n_clusters=2, random_state=0).fit(L)
labels = kmeans.labels_

# For each group estimate another regression line of Y on X
betas_group = []
for i in range(2):
    idx = labels == i
    model_group = sm.OLS(Y[idx], sm.add_constant(X[idx])).fit()
    betas_group.append(model_group.params)

# Plotting

marker_dict = {0: 'o', 1: 'x'}  # replace with your categories and markers
color_dict = {'category1': 'red', 'category2': 'blue'}  # replace with your categories and colors


Yint = (X**2 + UY + 3 + UY + X**2)/2
    

plt.figure(figsize=(10, 6))


# plt.scatter(X, Y, c=L.squeeze(), cmap='viridis')

for h in range(2):
    mask = H == h
    plt.scatter(X[mask], Y[mask], marker=marker_dict[h], c=L[mask].squeeze(), label=f'H={h}')

plt.scatter(X, Yint, c = 'red')
    
plt.plot(x_line, y_line_naive, color='red', label='Naive regression')
plt.plot(x_line, y_line_adjusted, color='blue', label='Adjusted regression')

for i in range(2):
    y_line_group = np.dot(x_line_naive, betas_group[i])
    plt.plot(x_line, y_line_group, color='blue', linestyle='dashed')

plt.colorbar(label='Confounder (L)')
plt.xlabel('Cause (X)')
plt.ylabel('Outcome (Y)')
plt.legend()
plt.title('Scatter plot of Y vs X colored by confounder L with regression lines')
plt.show()

# Make a table showing all X coefficients with suitable row names
data = {
    'Naive': [beta_naive[1]],
    'Adjusted': [beta_adjusted[1]],
    'Group 0': [betas_group[0][1]],
    'Group 1': [betas_group[1][1]],
    # 'Group 2': [betas_group[2][1]]
}
df = pd.DataFrame(data)
df.index = ['X coefficient']
print(df)

marker_dict = {0: 'o', 1: 'x'}  # replace with your categories and markers
color_dict = {'category1': 'red', 'category2': 'blue'}  # replace with your categories and colors

fig, axs = plt.subplots(1, 2, figsize=(10, 6))

# Left plot
for h in range(2):
    mask = H == h
    scatter = axs[0].scatter(X[mask], Y[mask], marker=marker_dict[h], c=L[mask].squeeze(), cmap='viridis', label=f'H={h}')

scatter = axs[0].scatter(X, Yint, c='red', edgecolors='black', label='Interventional')
    
axs[0].set_xlabel('Cause (X)')
axs[0].set_ylabel('Outcome (Y)')
axs[0].legend()
axs[0].set_title('Scatter plot of Y vs X colored by confounder L')

# Right plot
axs[1].hist(Y, bins=30, alpha=0.5, orientation='horizontal', label='Y observational')
axs[1].hist(Yint, bins=30, alpha=0.5, orientation='horizontal', label='Y interventional')
axs[1].set_ylabel('Outcome (Y)')
axs[1].set_xlabel('Frequency')
axs[1].legend(loc='upper right')
axs[1].set_title('Histogram of Yobservational and Yinterventional')

plt.tight_layout()

plt.savefig(plot_folder + "on_dim_example" + ".pdf", bbox_inches='tight', dpi = 100)

