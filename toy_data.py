import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

root_folder = 'C:/Users/pfbur/Box/projects/CFL-GIP/'
import os
os.chdir(root_folder + 'VaDE_code/Pytorch-VaDE')

results_folder = root_folder + 'results/'
plot_folder = results_folder + 'plots/'

# Number of samples
n_samples = 10000



## option 1 to generate H and L
# Define the joint probability matrix M for H and L
alpha = 0.8 # can be between 0 and 1
beta = alpha - 0.01
gamma = 0.4
M = np.array([[0.5 - (alpha/2), 0.5 - (1-alpha)/2],  # Probabilities for L=0
              [0.5 - (1-beta)/2, 0.5 - (beta/2)]]) # Probabilities for L=1

# M = np.array([[0.5 - (alpha/2), 0.5 - (1-alpha)/2 - gamma],  # Probabilities for L=0
#               [0.5 + (gamma/2) - (1-beta)/2, 0.5  + (gamma/2) - (beta/2)]]) # Probabilities for L=1

# M = np.array([[0.05, 0.35],  # Probabilities for L=0
#               [0.5, 0.1]]) # Probabilities for L=1

# Check that M is a valid joint probability matrix (i.e., all entries are non-negative and sum to 1)
assert np.all(M >= 0) and np.isclose(np.sum(M), 1), "Invalid joint probability matrix"

# Compute the marginal probabilities of L and H
P_L = np.sum(M, axis=1)
P_H = np.sum(M, axis=0)

# Sample H and L from the joint distribution
HL = np.random.choice([0, 1, 2, 3], size=n_samples, p=M.flatten())

# M flatten is P(H = 0 | L = 0), P(H = 1 | L = 0), P(H = 0 | L = 1), P(H = 1 | L = 1)

L, H = HL // 2, HL % 2
# Compute the mutual information
MI = 0
for l in range(M.shape[0]):
    for h in range(M.shape[1]):
        if M[l, h] > 0:  # To avoid division by zero and log of zero
            MI += M[l, h] * np.log(M[l, h] / (P_L[l] * P_H[h]))

print("Mutual Information:", MI)

# H causes X (high-dimensional continuous data)
# Simulate Gaussian Mixture data directly
xdim = 1  # or 1, or any other positive integer

X = np.zeros((n_samples, xdim))

H0loc = -2
H1loc = 2

if xdim > 1:
    cov = 0.5 * np.eye(xdim)  # Covariance matrix
    X[H == 0] = np.random.multivariate_normal(mean=[H0loc]*xdim, cov=cov, size=np.sum(H == 0))
    X[H == 1] = np.random.multivariate_normal(mean=[H1loc]*xdim, cov=cov, size=np.sum(H == 1))
else:
    X[H == 0, 0] = np.random.normal(loc=H0loc, scale=0.5, size=np.sum(H == 0))
    X[H == 1, 0] = np.random.normal(loc=H1loc, scale=0.5, size=np.sum(H == 1))

# L causes Y and X causes Y
UY = np.random.normal(0,.1, size=X.shape[0])
betaL = 3



Y = betaL * L + np.sum(X, axis=1) + UY

np.var(Y)

Yint = (betaL + np.sum(X, axis=1) + UY + np.sum(X, axis=1) + UY)/2


# Create a DataFrame for easier plotting
df_dict = {
    'H': H,
    'L': L,
    'HL': HL,
    'Y': Y,
    'Yint': Yint
}
for i in range(xdim):
    df_dict[f'X{i+1}'] = X[:, i]

df = pd.DataFrame(df_dict)


# scale data 

# # # Initialize the scaler
# scaler = StandardScaler()

# # Select only the columns to be scaled
# columns_to_scale = ['Y'] + [f'X{i+1}' for i in range(xdim)]
# df_to_scale = df[columns_to_scale]

# # Fit and transform the data
# df_scaled = pd.DataFrame(scaler.fit_transform(df_to_scale), columns=columns_to_scale)

# # Replace the original columns with the scaled ones
# df[columns_to_scale] = df_scaled


# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()

# # Select only the columns to be scaled
# columns_to_scale = ['Y'] + [f'X{i+1}' for i in range(xdim)]
# df_to_scale = df[columns_to_scale]

# # Fit and transform the data
# df_scaled = pd.DataFrame(scaler.fit_transform(df_to_scale), columns=columns_to_scale)

# # Replace the original columns with the scaled ones
# df[columns_to_scale] = df_scaled


print(df.head())


df.to_csv('toy_data.csv', index=False)


# take a sample for plotting
df_plot = df.sample(n=1000, random_state=1).reset_index(drop = True)


# marker_dict = {0: "s", 1: "X"} # replace with your actual team names and desired markers

# # Create a new column in df_plot for marker styles
# df_plot['marker'] = df_plot['L'].map(marker_dict)
# Create a PairGrid
sns.set(font_scale=2)
# # Create a pairplot of the data
sns.pairplot(df_plot, hue='L')#, plot_kws={style:'L', markers:['o', 's']})
plt.show()



# plot Y against X and obs interventional for X 1dim



if xdim == 1:
    marker_dict = {0: 'o', 1: 'x'}  # replace with your categories and markers
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    # Left plot
    for h in range(2):
        mask = df_plot["H"] == h
        scatter = axs[0].scatter(df_plot["X1"][mask], df_plot["Y"][mask], marker=marker_dict[h], c=df_plot["L"][mask].squeeze(), cmap = 'PuOr', label=f'H={h}')
    
    scatter = axs[0].scatter(df_plot["X1"], df_plot["Yint"], c='green', edgecolors='black', label='Interventional')
        
    axs[0].set_xlabel('Cause (X)')
    axs[0].set_ylabel('Outcome (Y)')
    axs[0].legend()
    axs[0].set_title('colored by confounder L')
    
    # Right plot
    axs[1].hist(df_plot["Y"], bins=30, alpha=0.5, orientation='horizontal', label='Y observational')
    axs[1].hist(df_plot["Yint"], bins=30, alpha=0.5, orientation='horizontal', label='Y interventional')
    axs[1].set_ylabel('Outcome (Y)')
    axs[1].set_xlabel('Frequency')
    axs[1].legend(loc='upper right')
    axs[1].set_title('')
    
    plt.tight_layout()
    
    plt.savefig(plot_folder + "on_dim_example" + ".pdf", bbox_inches='tight', dpi = 100)



#%% more plots


# plot for X 2dim
# Define a dictionary for markers
markers = {0: "o", 1: "^"}

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))


# Scatter plot 2: x1 vs x2 with color=y and marker=H
sns.scatterplot(x='X1', y='X2', hue='Y', style='H', data=df_plot, ax=axs[0], markers=markers)
axs[0].set_title('Scatter plot with marker=H')
# Scatter plot 1: x1 vs x2 with color=y and marker=L
sns.scatterplot(x='X1', y='X2', hue='Y', style='L', data=df_plot, ax=axs[1], markers=markers)
axs[1].set_title('Scatter plot with marker=L')

# Show the plots
plt.tight_layout()
plt.show()



# Create a new figure for the 3D plot
fig = plt.figure()
sns.set(font_scale=1)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Add the scatter plot data with different markers for H=0 and H=1
for h in [0, 1]:
    df_plot_h = df_plot[df_plot['H'] == h]
    ax.scatter(df_plot_h['X1'], df_plot_h['X2'], df_plot_h['Y'], c=df_plot_h['L'], cmap = 'RdBu', marker='o' if h == 0 else '^')

# Set the labels for the axes
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('marker = H, color = L')

# Change the orientation of the plot
ax.view_init(elev=15., azim=-75)

# Add a legend

plt.show()


#%%

plt.hist(df["Y"])


# to marginalize out L
# keep x, y etc fixed and compute a version of Y for each L = 0 and L = 1
Y_L0 = X[:,0] + X[:,1] + UY
Y_L1 = X[:,0] + X[:,1] + UY + 1

Y_wtL = P_L[0] * Y_L0 + P_L[1] * Y_L1
plt.hist(Y_wtL)

dfY = pd.DataFrame({
    'L0': Y_L0,
    'L1': Y_L1,
    'Y': Y,
    'Y_wtL': Y_wtL
})







# Create the histogram using Seaborn
sns.histplot(data=dfY.melt(), x='value', hue='variable', element='step', stat='density', common_norm=False)

plt.show()




###
# Assuming df is your DataFrame and "Y" is your column
fig, axs = plt.subplots()

# Create histogram for Y
axs.hist(df["Y"], bins=50, alpha=0.5, label='Y')

# Create histogram for Y where L==0
axs.hist(df[df["L"]==0]["Y"], bins=50, alpha=0.5, label='Y when L==0')

# Create histogram for Y where L==1
axs.hist(df[df["L"]==1]["Y"], bins=50, alpha=0.5, label='Y when L==1')

# Create histogram for Y where L==1
axs.hist(dfY["Y_wtL"], bins=50, alpha=0.5, label='YwtL')


axs.set_xlabel('Value')
axs.set_ylabel('Frequency')
axs.set_title('Histogram of Y, Y when L==0 and Y when L==1')
axs.legend()

plt.show()
###

# Assuming df is your DataFrame and "Y" is your column
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Create histogram for Y
axs[0].hist(df["Y"], bins=50, alpha=0.5, label='Y')

# Create histogram for Y where L==0
axs[0].hist(df[df["L"]==0]["Y"], bins=50, alpha=0.5, label='Y when L==0')

# Create histogram for Y where L==1
axs[0].hist(df[df["L"]==1]["Y"], bins=50, alpha=0.5, label='Y when L==1')

axs[0].set_xlabel('Value')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Histogram of Y, Y when L==0 and Y when L==1')
axs[0].legend()

# Get the current x-axis limit
x_limit = axs[0].get_xlim()

# Create histogram for Y_wtL in the second plot
axs[1].hist(dfY["Y_wtL"], bins=50, alpha=0.5, label='YwtL')

# Set the x-axis limit to be the same as the first plot
axs[1].set_xlim(x_limit)

axs[1].set_xlabel('Value')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Histogram of YwtL')
axs[1].legend()

plt.tight_layout()
plt.show()
