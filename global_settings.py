import torch.utils.data
from torchvision import datasets, transforms
dim_x = 2
dim_y = 1 # the original value

dim_x_and_y = dim_x + dim_y

in_dim_autoencoder = dim_x_and_y
# in_dim = 20 # 784
# encoder_units = [128,128,512]
encoder_units = [512, 512, 2048]
encoder_units = [7, 7, 9] # remember that they are doubled because we have two independent networks for X1 and X2.



latent_dim_x = dim_x
latent_dim_y = 3

n_classes = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# encoder_units = [int(0.66 * in_dim), int(0.66 * in_dim), int(2.7 * in_dim)]
# decoder_units = encoder_units.copy()
# decoder_units.reverse()
# can also specify decoder units independently of encoder_units