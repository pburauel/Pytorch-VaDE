#%%
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


from hsic_torch import *
# how to get the model params out?


df_org = pd.read_csv('toy_data.csv')



fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

# Plot the observational distribution of Y
axs[0].hist(df_org["Y"], bins=30, alpha=0.5)
axs[0].set_title('Observational')

# Plot the true interventional distribution of Y
axs[1].hist(df_org["Yint"], bins=30, alpha=0.5)
axs[1].set_title('True Interventional')


axs[1].set_xlabel('Y')


# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5)

plt.show()


df_obs = df_org[[col for col in df_org.columns if col.startswith('X') or col == 'Y']]





# feed through model
xy_hat, mu, log_var, z = vade.VaDE(torch.from_numpy(df_obs.values).float())




# np.cov(z.T)
# disentangle

# df_org = df_org.numpy()
xy_hat = xy_hat.detach().numpy()
z = z.detach().numpy()

xhat = xy_hat[:,:dim_x]
yhat = xy_hat[:,dim_x:]

z1 = z[:,:latent_dim_x]
z2 = z[:,latent_dim_x:]


# are Z1 and Z2 dependent?

## this HSIC implementation is not working, test statistic scales with number of observations!!

# nest step, try this: https://github.com/Black-Swan-ICL/PyRKHSstats/tree/main

# hsic_gam_torch(torch.from_numpy(z1), torch.from_numpy(z2))

# hsic_gam_torch(torch.from_numpy(z1[1:100]), torch.from_numpy(z2[1:100]))

# hsic_gam_torch(torch.from_numpy(z1[1:10]), torch.from_numpy(z2[1:10]))


# testStat, thresh = hsic_gam(torch.from_numpy(z1), torch.from_numpy(z2), alph = 0.05)
## this is computing a test stat but it doesnt produce a p value....

df_obs_z = pd.concat((pd.DataFrame(df_obs), pd.DataFrame(z)), axis = 1)

col_names = ["X"+str(i+1) for i in range(dim_x)] + ["Y"] + ["ZX"+str(i+1) for i in range(latent_dim_x)] + ["ZY"+str(i+1) for i in range(latent_dim_y)]


df_obs_z.columns = col_names

# how good is the reconstruction?
# X, Y against Xhat, Yhat
df_xy_xyhat = pd.concat((pd.DataFrame(df_obs), pd.DataFrame(xy_hat)), axis = 1)
df_xy_xyhat.columns = ["X"+str(i+1) for i in range(dim_x)] + ["Y"] + ["Xhat"+str(i+1) for i in range(latent_dim_x)] + ["Yhat"]

# Set the figure size
plt.figure(figsize=(3,3))  # You can adjust the values as per your requirement

# Your plot code here
sns.pairplot(df_xy_xyhat)

# Save the figure
plt.savefig(plot_folder + "model_" + time_str  + "_pairplot.pdf", bbox_inches='tight', dpi = 100)

# Get all columns that start with 'X'
x_cols = [col for col in df_org.columns if col.startswith('X')]
#%%% run regressions




# naive regression



# Add a constant to the independent values
X = sm.add_constant(df_org[x_cols])

# Fit the model
model1 = sm.OLS(df_org['Y'], X).fit()

# Print out the statistics
print(model1.summary())


## true model (with known confounder)
# Get all columns that start with 'X' and 'L'
x_l_cols = x_cols + [col for col in df_org.columns if col.startswith('L')]

# Add a constant to the independent values
X = sm.add_constant(df_org[x_l_cols])

# Fit the model
model2 = sm.OLS(df_org['Y'], X).fit()

# Print out the statistics
print(model2.summary())


## now estimate the model with recoverd confounder
# Get all columns that start with 'X' and 'L'
x_z_cols = x_cols + [col for col in df_obs_z.columns if col.startswith('ZY')]

# Add a constant to the independent values
X = sm.add_constant(df_obs_z[x_z_cols])

# Fit the model
model3 = sm.OLS(df_obs_z['Y'], X).fit()

# Print out the statistics
print(model3.summary())



#%% deconfound


pi_c = vade.VaDE.pi_prior.detach().numpy()
pi_c = np.clip(pi_c, a_min = 0, a_max = pi_c.max())
pi_c = pi_c/pi_c.sum()
mu_prior = vade.VaDE.mu_prior.detach().numpy() # size #C x #L
log_var_prior = vade.VaDE.log_var_prior.detach().numpy()
# var_prior = ###

# prior for Z2
# draw from categorical with proba pi_c
draws = np.random.choice(np.arange(len(pi_c)), size = z1.shape[0], p=pi_c)

# now use mu_prior and log_var_prior
var_prior = np.exp(log_var_prior)


# get the means and variances for each draw
means = mu_prior[draws]
variances = var_prior[draws]

# Generate samples
prior_z2 = np.random.normal(loc=means, scale=np.sqrt(variances))[:,dim_x:]

posterior_z1_x = z1

# feed through model


xy_decode = vade.VaDE.decode(torch.cat((torch.from_numpy(posterior_z1_x), torch.from_numpy(prior_z2).float()), axis = 1))
xy_decode = pd.DataFrame(xy_decode.detach().numpy())

# Get the number of 'X' columns
xdim = len([col for col in df_obs.columns if col.startswith('X')])

# Create a list of new column names
new_columns = [f'X{i+1}hat' for i in range(xdim)] + ['Yhat']

# Rename the columns
xy_decode.columns = new_columns
# run regression

# Add a constant to the independent values
X = sm.add_constant(xy_decode[[c + "hat" for c in x_cols]])

# Fit the model
modelz = sm.OLS(xy_decode['Yhat'], X).fit()

print(modelz.summary())

# sns.histplot(xy_decode[,2])
# sns.histplot(y_decode.detach().numpy()[:,2])
# sns.histplot(yhat[:,0])
# sns.pairplot(xy_decode)

df_org_decode = pd.concat((df_obs, xy_decode), axis = 1)

sns.pairplot(df_org_decode)


# plot three histograms in one plot
# 1: observational distribution of Y: df_obs["Y"]
# 2: true interventiaonal distribution of Y: df_Yint["Yint"]
# 3: estimated interventional distribution of Y: xy_decode['Yhat']


fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)

# Plot the observational distribution of Y
axs[0].hist(df_obs["Y"], bins=30, alpha=0.5)
axs[0].set_title('Observational')

# Plot the true interventional distribution of Y
axs[1].hist(df_org["Yint"], bins=30, alpha=0.5)
axs[1].set_title('True Interventional')

# Plot the estimated interventional distribution of Y
axs[2].hist(xy_decode['Yhat'], bins=30, alpha=0.5)
axs[2].set_title('Estimated Interventional')
axs[2].set_xlabel('Y')


# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5)

plt.show()
fig.savefig(plot_folder + "model_" + time_str  + "_Yhistograms.pdf", 
            bbox_inches='tight',
            dpi = 333)   # save the figure to file     


#%% conditional interventional distribution





#%%





# # Assuming df_org is your DataFrame
# df_conditional = df_org.copy()
# df_conditional['X1_bin'] = pd.cut(df_conditional['X1'], bins=10)
# df_conditional['X2_bin'] = pd.cut(df_conditional['X2'], bins=10)


# # Group by 'X1_bin', 'X2_bin', and calculate the counts for L=0 and L=1
# counts_L0 = df_conditional[df_conditional['L'] == 0].groupby(['X1_bin', 'X2_bin']).size()
# counts_L1 = df_conditional[df_conditional['L'] == 1].groupby(['X1_bin', 'X2_bin']).size()

# # Calculate the total number of observations in each X1-X2 bin group
# total_counts = df_conditional.groupby(['X1_bin', 'X2_bin']).size()

# # Compute the marginal probabilities of L=0 and L=1 in each X1-X2 bin group
# pL0 = counts_L0 / total_counts
# pL1 = counts_L1 / total_counts

# # Fill NaN values with 0 (in case there are bins with no zeros or ones)
# pL0.fillna(0, inplace=True)
# pL1.fillna(0, inplace=True)

# # Create a DataFrame from pL0 and pL1
# df_marginal_probabilities = pd.DataFrame({'pL0': pL0, 'pL1': pL1}).reset_index()

# df_merged = pd.merge(df_conditional, df_marginal_probabilities, on=['X1_bin', 'X2_bin'])
# # Create a new column that is equal to pL0 if L = 0 and pL1 if L = 1
# df_merged['marginal_proba'] = np.where(df_merged['L'] == 0, df_merged['pL0'], df_merged['pL1'])

# df_merged

# # For each X1_bin X2_bin group, compute a weighted mean of Y where the weight is equal to marginal_proba
# # For each X1_bin X2_bin group, compute a weighted mean of Y where the weight is equal to marginal_proba
# df_weighted_avg = df_merged.groupby(['X1_bin', 'X2_bin']).apply(lambda x: np.average(x['Y'], weights=x['marginal_proba'])).reset_index(name='Y_weighted_avg')



# # Group by 'X1_bin' and 'X2_bin', and calculate the average of 'Y'
# df_grouped = df_conditional.groupby(['X1_bin', 'X2_bin'])['Y'].mean().reset_index(name='Y_avg')
# #@ change this so that all columnd of df_conditional will be in df_grouped


# # Pivot the DataFrame to create a matrix for the heatmap
# df_pivot = df_grouped.pivot('X1_bin', 'X2_bin', 'Y_avg')

# # Plot the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(df_pivot, cmap='viridis')
# plt.title('Heatmap of Average Y values for X1 and X2 bins')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.show()
