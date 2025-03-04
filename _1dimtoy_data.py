import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler


# Number of samples
n_samples = 10000



## option 1 to generate H and L
# Define the joint probability matrix M for H and L
alpha = 0.8 # can be between 0 and 1
beta = alpha - 0.01
gamma = 0.1
M = np.array([[0.5 - (alpha/2), 0.5 - (1-alpha)/2],  # Probabilities for L=0
              [0.5 - (1-beta)/2, 0.5 - (beta/2)]]) # Probabilities for L=1

M = np.array([[0.5 - (alpha/2), 0.5 - (1-alpha)/2 - gamma],  # Probabilities for L=0
              [0.5 + (gamma/2) - (1-beta)/2, 0.5  + (gamma/2) - (beta/2)]]) # Probabilities for L=1



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
X = np.zeros((n_samples,1))
X[H == 0] = np.random.normal(loc=[-1], scale=[0.5], size=(np.sum(H == 0), 1))
X[H == 1] = np.random.normal(loc=[1], scale=[0.5], size=(np.sum(H == 1), 1))

# L causes Y and X causes Y
# L causes Y and X causes Y
UY = np.random.normal(0,.1, size = X.shape[0])


# Create a copy of L
scale = np.copy(L)

scale = np.where(scale==0, 0.5, scale)
scale = np.where(scale==1, 1.5, scale)

# Multiply X by scale, reshape X, and add L and UY
Y = L + scale * X.reshape(-1,) + UY

# Create a DataFrame for easier plotting
df = pd.DataFrame({
    'H': H,
    'L': L,
    'HL': HL,
    'X': X.reshape(-1,),  
    'Y': Y
})

# scale data 

# # Initialize the scaler
# scaler = StandardScaler()

# # Select only the columns to be scaled
# columns_to_scale = ['X1', 'X2', 'Y']
# df_to_scale = df[columns_to_scale]

# # Fit and transform the data
# df_scaled = pd.DataFrame(scaler.fit_transform(df_to_scale), columns=columns_to_scale)

# # Replace the original columns with the scaled ones
# df[columns_to_scale] = df_scaled

# Y_transformed = df['Y']


print(df.head())


# df.to_csv('toy_data.csv', index=False)

# take a sample for plotting
df_plot = df.sample(n=1000, random_state=1)


# marker_dict = {0: "s", 1: "X"} # replace with your actual team names and desired markers

# # Create a new column in df_plot for marker styles
# df_plot['marker'] = df_plot['L'].map(marker_dict)
# Create a PairGrid
sns.set(font_scale=2)
# # Create a pairplot of the data
sns.pairplot(df_plot, hue='L')#, plot_kws={style:'L', markers:['o', 's']})
plt.show()



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
ax.view_init(elev=45., azim=-75)

# Add a legend

plt.show()
