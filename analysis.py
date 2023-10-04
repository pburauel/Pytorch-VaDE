#%%
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


from hsic_torch import *
# how to get the model params out?


df = pd.read_csv('toy_data.csv')

df.head()


df_org = df[["X1", "X2", "Y"]]
df_org = torch.from_numpy(df_org.values)
df_org = df_org.float()


# feed through model

xy_hat, mu, log_var, z = vade.VaDE(df_org)

pi_c = vade.VaDE.pi_prior.detach().numpy()
pi_c = pi_c/pi_c.sum()


# #compute C
# gamma = vade.compute_gamma(torch.from_numpy(z), vade.VaDE.pi_prior)
# pred = torch.argmax(gamma, dim=1)

# np.cov(z.T)
# disentangle
df_org = df_org.numpy()
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

df_org_z = pd.concat((pd.DataFrame(df_org), pd.DataFrame(z)), axis = 1)

col_names = ["X"+str(i+1) for i in range(dim_x)] + ["Y"] + ["ZX"+str(i+1) for i in range(latent_dim_x)] + ["ZY"+str(i+1) for i in range(latent_dim_y)]


df_org_z.columns = col_names

# how good is the reconstruction?
# X, Y against Xhat, Yhat
df_xy_xyhat = pd.concat((pd.DataFrame(df_org), pd.DataFrame(xy_hat)), axis = 1)
df_xy_xyhat.columns = ["X"+str(i+1) for i in range(dim_x)] + ["Y"] + ["Xhat"+str(i+1) for i in range(latent_dim_x)] + ["Yhat"]

sns.pairplot(df_xy_xyhat)


#%%%
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



#%%

#%% prior of Z2

mu_prior = vade.VaDE.mu_prior # size #C x #L
log_var_prior = vade.VaDE.log_var_prior
# var_prior = ###


# draw from categorical with proba pi_c

draws = np.random.choice(np.arange(len(pi_c)), size = z1.shape[0], p=pi_c)

# now use this to draw


# feed through model
posterior_z1_x = z1
prior_z2 = np.random.normal(size=z2.shape)
prior_z2 = torch.from_numpy(prior_z2).float()
vade.VaDE.eval()
y_decode = vade.VaDE.decode(torch.cat((torch.from_numpy(posterior_z1_x), prior_z2), axis = 1))

sns.histplot(y_decode.detach().numpy()[:,2])
sns.histplot(y_decode.detach().numpy()[:,2])
sns.histplot(yhat[:,0])

# use mu and logvar to create prior_z2


std = torch.exp(log_var/2)[:,dim_x:]
mu_z2 = mu[:,dim_x:]
eps = torch.randn_like(std)
prior_z2 = mu_z2 + eps * std

y_decode = vade.VaDE.decode(torch.cat((torch.from_numpy(posterior_z1_x), prior_z2), axis = 1))
sns.histplot(y_decode.detach().numpy()[:,2])

sns.histplot(prior_z2.detach().numpy()[:,1])

        


