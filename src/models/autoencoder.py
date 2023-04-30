import torch 
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()

class Autoencoder(nn.Module):
    """ The autoencoder is a combination of the encoder and decoder
    
    Args:
        encoder (nn.Module): The encoder to use
        decoder (nn.Module): The decoder to use
        latent_dims (int): The number of latent dimensions to use
    """
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        latent_dims = 2
        encoder = nn.Sequential(
                    nn.Linear(in_features=500, out_features=128),
                    nn.ReLU(),
                    # bottleneck layer
                    nn.Linear(in_features=128, out_features=latent_dims)
                ).to(device)

        decoder = nn.Sequential(
                    nn.Linear(in_features=latent_dims, out_features=128),
                    nn.ReLU(),
                    nn.Linear(in_features=128, out_features=500),
                    # nn.Sigmoid()
                ).to(device)
        self.encoder = encoder
        self.decoder = decoder

        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)