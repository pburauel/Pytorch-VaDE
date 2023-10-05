#%%
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


from hsic_torch import *
# how to get the model params out?


df_org = pd.read_csv('toy_data.csv')


df_obs = df_org[["X1", "X2", "Y"]]
df_obs = torch.from_numpy(df_obs.values)
df_obs = df_obs.float()


# feed through model
xy_hat, mu, log_var, z = vade.VaDE(df_obs)




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


#%%% run regressions
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



#%% deconfound


pi_c = vade.VaDE.pi_prior.detach().numpy()
pi_c = pi_c/pi_c.sum()
mu_prior = vade.VaDE.mu_prior.detach().numpy() # size #C x #L
log_var_prior = vade.VaDE.log_var_prior.detach().numpy()
# var_prior = ###

# prior for Z2
# draw from categorical with proba pi_c
draws = np.random.choice(np.arange(len(pi_c)), size = z1.shape[0], p=pi_c)

# now use mu_prior and log_var_prior
var_prior = np.exp(log_var_prior)


# Use advanced indexing to get the means and variances for each draw
means = mu_prior[draws]
variances = var_prior[draws]

# Generate samples
prior_z2 = np.random.normal(loc=means, scale=np.sqrt(variances))[:,dim_x:]

posterior_z1_x = z1

# feed through model


xy_decode = vade.VaDE.decode(torch.cat((torch.from_numpy(posterior_z1_x), torch.from_numpy(prior_z2).float()), axis = 1))
xy_decode = xy_decode.detach().numpy()


sns.histplot(xy_decode.detach().numpy()[:,2])
sns.histplot(y_decode.detach().numpy()[:,2])
sns.histplot(yhat[:,0])

