import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
import pandas as pd

# Set the number of samples n
n = 10000

# Generate a one-dimensional confounder H
H = np.random.normal(size=(n, 1))

# Generate cause X, which is influenced by H
X = H + np.random.normal(size=(n, 1)) * 2

# Generate outcome Y, which is a function of X and H
Y = X + 3 * H + np.random.normal(size=(n, 1)) * .1

# Fit a naive linear regression model of Y on X
X_naive = sm.add_constant(X)
model_naive = sm.OLS(Y, X_naive).fit()

# Fit a linear regression model of Y on X and H
XH = sm.add_constant(np.concatenate([X, H], axis=1))
model_adjusted = sm.OLS(Y, XH).fit()

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
kmeans = KMeans(n_clusters=3, random_state=0).fit(H)
labels = kmeans.labels_

# For each group estimate another regression line of Y on X
betas_group = []
for i in range(3):
    idx = labels == i
    model_group = sm.OLS(Y[idx], sm.add_constant(X[idx])).fit()
    betas_group.append(model_group.params)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, c=H.squeeze(), cmap='viridis')
plt.plot(x_line, y_line_naive, color='red', label='Naive regression')
plt.plot(x_line, y_line_adjusted, color='blue', label='Adjusted regression')

for i in range(3):
    y_line_group = np.dot(x_line_naive, betas_group[i])
    plt.plot(x_line, y_line_group, color='blue', linestyle='dashed')

plt.colorbar(label='Confounder (H)')
plt.xlabel('Cause (X)')
plt.ylabel('Outcome (Y)')
plt.legend()
plt.title('Scatter plot of Y vs X colored by confounder H with regression lines')
plt.show()

# Make a table showing all X coefficients with suitable row names
data = {
    'Naive': [beta_naive[1]],
    'Adjusted': [beta_adjusted[1]],
    'Group 0': [betas_group[0][1]],
    'Group 1': [betas_group[1][1]],
    'Group 2': [betas_group[2][1]]
}
df = pd.DataFrame(data)
df.index = ['X coefficient']
print(df)
