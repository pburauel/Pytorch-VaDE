#%%
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle

from hsic_torch import *
# how to get the model params out?

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


def fn_plot_hist_by_X(data, snippet):
    # data must be n x 2 data frame, first column: X, second column Y
    
    # Generate labels like "1 (low X)" to "10 (high X)"
    bin_labels = [f"{i+1} {'(low X)' if i == 0 else ('(high X)' if i == 9 else '')}" for i in range(10)]
    
    # Define bin edges
    bin_edges = np.array([10, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0])
    
    # Setting up the figure and axes
    fig, axs = plt.subplots(10, 1, figsize=(10, 20), sharex=True)
    
    colors = sns.color_palette("cubehelix", 10)
    
    # Loop through each bin to plot the histograms
    for i in range(10):
        upper_bin_edge = bin_edges[i]
        lower_bin_edge = bin_edges[i+1]
        subset = data[(data['X'] >= lower_bin_edge) & (data['X'] < upper_bin_edge)]
        axs[i].hist(subset['Y'], bins=30, color=colors[i], edgecolor='black')
        axs[i].set_title(bin_labels[i])
        axs[i].set_ylabel('Frequency')  
        axs[i].set_title(f"Bin {i+1} ({lower_bin_edge:.2f} <= X < {upper_bin_edge:.2f})")

        
    # Set a common x-label
    axs[-1].set_xlabel('Y Value')
    axs[-1].set_xlim([-.2, 1.2])
    
    
    # Add overall title to the figure
    fig.suptitle(snippet, y=1.02)

    
    plt.tight_layout()
    # plt.show()
    
    plt.savefig(plot_folder + "model" + time_str + "hist_deconf_" + snippet + ".pdf", bbox_inches='tight', dpi = 100)


def fn_plot_hist_by_X_multiple(df_dict, time_str, snippet, density = True):
    # Generate labels like "1 (low X)" to "10 (high X)"
    bin_labels = [f"{i+1} {'(low X)' if i == 0 else ('(high X)' if i == 9 else '')}" for i in range(10)]
    
    # Setting up the figure and axes
    fig, axs = plt.subplots(10, 1, figsize=(10, 20), sharex=True)
    
    # Assuming all DataFrames in df_dict have the same range of 'X', using the first one to get bin edges
    # _, bin_edges = pd.cut(df_dict[list(df_dict.keys())[0]]['X'], bins=10, retbins=True)
    bin_edges = np.array([10, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0])
    # Colors (you can adjust this if you have more than 10 datasets in df_dict)
    colors = sns.color_palette("tab10", len(df_dict))
    
    # Loop through each bin to plot the histograms
    for i in range(10):
        upper_bin_edge = bin_edges[i]
        lower_bin_edge = bin_edges[i+1]
        
        for j, (key, df) in enumerate(df_dict.items()):
            subset = df[(df['X'] >= lower_bin_edge) & (df['X'] < upper_bin_edge)]
            axs[i].hist(subset['Y'], bins=30, color=colors[j], alpha=0.5, edgecolor='black', label=key, density=density)
            axs[i].set_title(f"Bin {i+1} ({lower_bin_edge:.2f} <= X < {upper_bin_edge:.2f})")
            axs[i].set_ylabel('Density' if density == True else "Frequency")
            if i == 0:
                axs[i].legend()
    
    # Set a common x-label and x-limits
    axs[-1].set_xlabel('Y Value')
    axs[-1].set_xlim([-.2, 1.2])
    
    # Adjust layout
    plt.tight_layout()
    fig.suptitle("Hist by X Bins", y=1.02)
    
    # Save the figure
    snippet = snippet + "_dens" if density == True else "_freq" 
    plt.savefig(plot_folder + "model" + time_str + "_histogram_" + snippet + ".pdf", bbox_inches='tight', dpi=100)



# Load the scaler from the saved file
with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)


# load original data
df_org = pd.read_csv('toy_data.csv')


# Get all columns that start with 'X'
x_cols = [col for col in df_org.columns if col.startswith('X')]

# create observed data dataframe
df_obs = df_org[x_cols + ['Y']]

# scale the observed data (as is done in the get_toy_data dataloader)
df_obs_sc = df_obs
df_obs_sc[["X1", "Y"]] = scaler.fit_transform(df_obs_sc[["X1", "Y"]])
# plot
fn_plot_hist_by_X(pd.DataFrame(df_obs_sc.rename(columns={"X1":"X"})), "orginal_data_confounded")


# scale the true interventional Yint using the same scaler that is used in get_toy_data dataloader
df_org_sc = df_org[["X1", "Yint"]]
df_org_sc[["X1", "Yint"]] = loaded_scaler.transform(df_org_sc[["X1", "Yint"]])
# plot
df_org_int_plot = pd.DataFrame(df_org_sc[["X1", "Yint"]]).rename(columns={"X1":"X", "Yint": 'Y'})
if dim_x == 1: fn_plot_hist_by_X(df_org_int_plot, "true_interventional")



# feed through model
vade.VaDE.eval()
with torch.no_grad():
    xy_hat, mu, log_var, z = vade.VaDE.forward(torch.from_numpy(df_obs.values).float())

t1 = xy_hat[:, 0].detach().numpy().reshape(-1,1).copy()
t2 = np.array(df_obs[["X1"]]).copy()
diff = abs(t1 - t2)
condition = diff <= 0.1

# Count the number of differences not more than 0.01
count = np.sum(condition == True)

# Get the indices of the rows that meet the condition
indices = condition == 1

print("Count:", count)

plt.scatter(t1, t2)
# plt.scatter(t1[indices], t2[indices], c = 'red')


### feed scaled data
# feed through model
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0, 1))

# # Fit and transform the data
# scaled_data = scaler.fit_transform(df_obs.values)
# # scaled_data = df_obs.values
# plt.hist(df_obs.values[:,0], bins = 30)
# plt.hist(scaled_data[:,0], bins = 30)


# vade.VaDE.eval()
# with torch.no_grad():
#     xy_hat_sc, _, _, _ = vade.VaDE.forward(torch.from_numpy(scaled_data).float())

# t1 = xy_hat_sc[:, 0].detach().numpy().reshape(-1,1).copy()
# t2 = np.array(scaled_data[:,0]).copy()
# plt.scatter(t1, t2)
# plt.scatter(t1[indices], t2[indices], c = 'red')



# vade.VaDE.eval()
# with torch.no_grad():
#     for _ in range(100):
#         xy_hat, mu, log_var, z = vade.VaDE(torch.from_numpy(df_obs.values).float())
#         xy_hat_accum += xy_hat
#         mu_accum += mu
#         log_var_accum += log_var
#         z_accum += z

# # Average the accumulators
# xy_hat = xy_hat_accum / 100
# mu = mu_accum / 100
# log_var = log_var_accum / 100
# z = z_accum / 100



# np.cov(z.T)
# disentangle

# df_org = df_org.numpy()
xy_hat = xy_hat.detach().numpy()

xy_hat_inverted = loaded_scaler.inverse_transform(xy_hat)


z = z.detach().numpy()

xhat = xy_hat[:,:dim_x]
yhat = xy_hat[:,dim_x:]

z1 = z[:,:latent_dim_x]
z2 = z[:,latent_dim_x:]

# plt.hist(z1)
# plt.hist(log_var.detach().numpy())

#%% check_VAE

vade.autoencoder.eval()
pred_ae = vade.autoencoder.forward(torch.from_numpy(df_obs.values).float())

t1 = df_obs.values[:,1]
t2 = pred_ae[:,1]
plt.scatter(t1, t2.detach().numpy())

plt.scatter(t1[indices], t2.detach().numpy()[indices], c = 'red')
#%% pairplots
# are Z1 and Z2 dependent?

## hsic_gam_torch HSIC implementation is not working, test statistic scales with number of observations!!
# nest step, try this: https://github.com/Black-Swan-ICL/PyRKHSstats/tree/main
# hsic_gam_torch(torch.from_numpy(z1), torch.from_numpy(z2))
# hsic_gam_torch(torch.from_numpy(z1[1:100]), torch.from_numpy(z2[1:100]))
# hsic_gam_torch(torch.from_numpy(z1[1:10]), torch.from_numpy(z2[1:10]))
# testStat, thresh = hsic_gam(torch.from_numpy(z1), torch.from_numpy(z2), alph = 0.05)

df_obs_z = pd.concat((pd.DataFrame(df_obs), pd.DataFrame(z)), axis = 1)

col_names = ["X"+str(i+1) for i in range(dim_x)] + ["Y"] + ["ZX"+str(i+1) for i in range(latent_dim_x)] + ["ZY"+str(i+1) for i in range(latent_dim_y)]


df_obs_z.columns = col_names

# how good is the reconstruction?
# X, Y against Xhat, Yhat
df_xy_xyhat = pd.concat((pd.DataFrame(df_obs), pd.DataFrame(xy_hat)), axis = 1)
df_xy_xyhat.columns = ["X"+str(i+1) for i in range(dim_x)] + ["Y"] + ["Xhat"+str(i+1) for i in range(dim_x)] + ["Yhat"]

# Set the figure size
plt.figure(figsize=(3,3))  # You can adjust the values as per your requirement
sns.pairplot(df_xy_xyhat)
plt.savefig(plot_folder + "model_" + time_str  + "_pairplot.pdf", bbox_inches='tight', dpi = 100)



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

# get pi_c and mu, var of latent from the vade model
vade.VaDE.eval()
with torch.no_grad():
    pi_c = vade.VaDE.pi_prior.detach().numpy()
    pi_c = np.exp(pi_c)#, a_min = 0, a_max = pi_c.max())    
    pi_c = pi_c/pi_c.sum()
    mu_prior = vade.VaDE.mu_prior.detach().numpy() # size #C x #L
    log_var_prior = vade.VaDE.log_var_prior.detach().numpy()
    var_prior = np.exp(log_var_prior)

##############################################################################################################
# 1: sample new data, use observed X, ie. use Z1|X but sample Z2 from its prior
##############################################################################################################
# prior for Z2
# draw from categorical with proba pi_c
draws = np.random.choice(np.arange(len(pi_c)), size = z1.shape[0], p = pi_c)

# get the means and variances for each draw
means = mu_prior[draws]
variances = var_prior[draws]

# Generate samples
prior_z2 = np.random.normal(loc=means, scale=np.sqrt(variances))[:,dim_x:]

posterior_z1_x = z1

# feed through model

vade.VaDE.eval()
with torch.no_grad():
    xy_decode_deconf_one_shot = vade.VaDE.decode(torch.cat((torch.from_numpy(posterior_z1_x), torch.from_numpy(prior_z2).float()), axis = 1))
xy_decode_deconf_one_shot = pd.DataFrame(xy_decode_deconf_one_shot.detach().numpy())

fn_plot_hist_by_X(pd.DataFrame(xy_decode_deconf_one_shot).rename(columns={0:"X", 1:"Y"}), "deconfounding_by_using_prior_Z2_one_shot")

##############################################################################################################
# 2: deconfound using Z1|X and sampling Z2 from its prior
##############################################################################################################
# prior for Z2
# draw from categorical with proba pi_c
z1_samples = 10
z2_samples = 100 # z2 samples per z1 sample

def generate_prior_z2():
    draws = np.random.choice(np.arange(len(pi_c)), size=z1.shape[0], p=pi_c)
    means = mu_prior[draws]
    variances = var_prior[draws]
    return np.random.normal(loc=means, scale=np.sqrt(variances))[:, dim_x:]


def generate_posterior_z1():
    vade.VaDE.eval()
    with torch.no_grad():
        _, _, _, z = vade.VaDE.forward(torch.from_numpy(df_obs_sc.values).float())
    z1 = z[:,:latent_dim_x]
    return z1.detach().numpy()


# feed through model
# for each posterior_z1
all_draws = []
for i_post_z1 in range(z1_samples):
    posterior_z1 = generate_posterior_z1()
    y_draws = []
    for j_prior_z2 in range(z2_samples):
        prior_z2 = generate_prior_z2()
        
        input_to_decoder = torch.cat((torch.from_numpy(posterior_z1), torch.from_numpy(prior_z2).float()), axis = 1)
        vade.VaDE.eval()
        with torch.no_grad():
            xy_decode_deconf = vade.VaDE.decode(input_to_decoder)
        
        xy_decode_deconf = pd.DataFrame(xy_decode_deconf.detach().numpy())

        # Convert the decoded results back to numpy for processing
        xy_decode_deconf_np = xy_decode_deconf.to_numpy()
        # invert scaling
        # xy_decode_deconf_np = loaded_scaler.inverse_transform(xy_decode_deconf_np)

        
        # First column remains the same
        x_part = xy_decode_deconf_np[:, :dim_x] # Taking every factor_z2_samples-th value from the 1st column

        # For the second column, reshape and then compute the mean along axis 1
        y_part = xy_decode_deconf_np[:, -1:]#.reshape(factor_z2_samples, posterior_z1_x.shape[0])
        y_draws.append(y_part)
    y_draws_np = np.array(y_draws)
    y_mean = np.mean(y_draws, axis = 0)
    
    xy_decode_deconf = np.column_stack((x_part, y_mean))
    all_draws.append(xy_decode_deconf)

reshaped_draws = np.array(all_draws).reshape(-1, 2)

dfplot = pd.DataFrame(reshaped_draws).rename(columns = {0: "X", 1: "Y"})

if dim_x == 1: fn_plot_hist_by_X(dfplot.copy(), "deconfounding_by_using_prior_Z2")


##############################################################################################################
# 3: sample new data, sample Z1 and Z2 from their priors
##############################################################################################################
# prior for Z2
# draw from categorical with proba pi_c
no_draws = z1.shape[0] * 2
draws = np.random.choice(np.arange(len(pi_c)), size = no_draws, p=pi_c)

# get the means and variances for each draw
means = mu_prior[draws]
variances = var_prior[draws]

# Generate samples
prior_z = np.random.normal(loc=means, scale=np.sqrt(variances))


# feed through model
vade.VaDE.eval()
with torch.no_grad():
    xy_decode_new_samples = vade.VaDE.decode(torch.from_numpy(prior_z).float())
xy_decode_new_samples = pd.DataFrame(xy_decode_new_samples.detach().numpy())


fn_plot_hist_by_X(pd.DataFrame(xy_decode_new_samples).rename(columns={0:"X", 1:"Y"}), "both_Z1_and_Z2_from_priors")


##############################################################################################################
#### make plots
##############################################################################################################

df_dict = {"true_interventional": df_org_int_plot, 
           "deconfounding_by_using_prior_Z2": dfplot,
           "orginal_data_confounded": pd.DataFrame(df_obs_sc.rename(columns={"X1":"X"}))}
fn_plot_hist_by_X_multiple(df_dict, time_str, snippet="both_interv", density = True)


df_dict = {"both_Z1_and_Z2_from_priors": pd.DataFrame(xy_decode_new_samples).rename(columns={0:"X", 1:"Y"}),
           "orginal_data_confounded": pd.DataFrame(df_obs_sc.rename(columns={"X1":"X"}))}
fn_plot_hist_by_X_multiple(df_dict, time_str, snippet="orig_and_new_data", density = True)



#%% snippets following
# # Get the number of 'X' columns
# xdim = len([col for col in df_obs.columns if col.startswith('X')])

# # Create a list of new column names
# new_columns = [f'X{i+1}hat' for i in range(xdim)] + ['Yhat']

# # Rename the columns
# xy_decode.columns = new_columns
# # run regression

# # Add a constant to the independent values
# X = sm.add_constant(xy_decode[[c + "hat" for c in x_cols]])

# # Fit the model
# modelz = sm.OLS(xy_decode['Yhat'], X).fit()

# print(modelz.summary())

# # sns.histplot(xy_decode[,2])
# # sns.histplot(y_decode.detach().numpy()[:,2])
# # sns.histplot(yhat[:,0])
# # sns.pairplot(xy_decode)

# df_org_decode = pd.concat((df_obs, xy_decode), axis = 1)

# sns.pairplot(df_org_decode)


# # plot three histograms in one plot
# # 1: observational distribution of Y: df_obs["Y"]
# # 2: true interventiaonal distribution of Y: df_Yint["Yint"]
# # 3: estimated interventional distribution of Y: xy_decode['Yhat']


# fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)

# # Plot the observational distribution of Y
# axs[0].hist(df_obs["Y"], bins=30, alpha=0.5)
# axs[0].set_title('Observational')

# # Plot the true interventional distribution of Y
# axs[1].hist(df_org["Yint"], bins=30, alpha=0.5)
# axs[1].set_title('True Interventional')

# # Plot the estimated interventional distribution of Y
# axs[2].hist(xy_decode['Yhat'], bins=30, alpha=0.5)
# axs[2].set_title('Estimated Interventional')
# axs[2].set_xlabel('Y')


# # Adjust the spacing between subplots
# plt.subplots_adjust(hspace=0.5)

# plt.show()
# fig.savefig(plot_folder + "model_" + time_str  + "_Yhistograms.pdf", 
#             bbox_inches='tight',
#             dpi = 333)   # save the figure to file     


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
