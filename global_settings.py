import torch.utils.data
from torchvision import datasets, transforms
dim_x_and_y = 3
noX = 2
in_dim = dim_x_and_y - noX # the original value
in_dim_autoencoder = dim_x_and_y
# in_dim = 20 # 784
# encoder_units = [128,128,512]
encoder_units = [512, 512, 2048]
encoder_units = [4, 4, 8]

latent_dim = 2
n_classes = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# encoder_units = [int(0.66 * in_dim), int(0.66 * in_dim), int(2.7 * in_dim)]
# decoder_units = encoder_units.copy()
# decoder_units.reverse()
# can also specify decoder units independently of encoder_units