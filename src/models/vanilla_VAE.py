import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.utils
import torch.distributions
import math
import numpy as np
from torch import Tensor
from torch.distributions import Distribution, Normal
from src.models.reparameterized_gaussian import ReparameterizedDiagonalGaussian

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

        self.encoder1 = nn.Linear(in_features=500, out_features=128).to(device)
        self.encoder2 = nn.ReLU().to(device)
        self.encoder_mu = nn.Linear(in_features=128, out_features=latent_dims).to(device)
        self.encoder_logvar = nn.Linear(in_features=128, out_features=latent_dims).to(device)


        self.decoder1 = nn.Linear(in_features=latent_dims, out_features=128).to(device)
        self.decoder2 = nn.ReLU().to(device)
        self.decoder3 = nn.Linear(in_features=128, out_features=500).to(device)


    def encode(self, x):
        h1 = self.encoder1(x)
        h2 = self.encoder2(h1)
        mu = self.encoder_mu(h2)
        logvar = self.encoder_logvar(h2)
        # torch rsample
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std

        self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        return z, mu, logvar     
    
    def decode(self, z):
        h1 = self.decoder1(z)
        h2 = self.decoder2(h1)
        return self.decoder3(h2)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return self.decode(z)
    
    
class VariationalAutoencoder(nn.Module):

    """ The autoencoder is a combination of the encoder and decoder
    
    Args:
        encoder (nn.Module): The encoder to use
        decoder (nn.Module): The decoder to use
        latent_dims (int): The number of latent dimensions to use
    """
    def __init__(self, input_shape, latent_dims, batch_size=100, decoder_type="alpha"):
        super(VariationalAutoencoder, self).__init__()

        self.input_shape = input_shape
        self.latent_features = latent_dims
        self.observation_features = np.prod(input_shape)
        self.batch_size = batch_size

        self.encoder1 = nn.Linear(in_features=500, out_features=128).to(device)
        self.encoder2 = nn.ReLU().to(device)
        self.encoder_logmu = nn.Linear(in_features=128, out_features=latent_dims).to(device)
        self.encoder_logvar = nn.Linear(in_features=128, out_features=latent_dims).to(device)

        self.decoder1 = nn.Linear(in_features=latent_dims, out_features=128).to(device)
        self.decoder2 = nn.ReLU().to(device)
        self.decoder3 = nn.Linear(in_features=128, out_features=500).to(device)

        self.kl = 0

        self.decoder_type = decoder_type

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_dims])))
        
    def _encode(self, x):
        h1 = self.encoder1(x)
        h2 = self.encoder2(h1)
        log_mu = self.encoder_logmu(h2)
        mu = torch.exp(log_mu)
        log_var = self.encoder_logvar(h2)
        return mu, log_var

    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        mu, log_var = self._encode(x)
        sigma = torch.exp(0.5 * log_var)
        log_sigma = torch.log(sigma)
        
        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        px_logits = self.decode(z)
        px_logits = px_logits.view(-1, *self.input_shape) # reshape the output

        # Gaussian observation model
        return Normal(loc=px_logits, scale=0.5)

        # return Bernoulli(logits=px_logits, validate_args=False)

    def encode(self, x):
        mu, log_var = self._encode(x)
        # torch rsample
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        z = mu + eps*std
        
        x = x.view(x.size(0), -1)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        self.qz = self.posterior(x)
        
        # define the prior p(z)
        self.pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = self.qz.rsample()
        
        # define the observation model p(x|z) = N(x | 0.25)
        self.px = self.observation_model(z)

        self.kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

        return z, mu, log_var     
    

    def decode(self, z):
        h1 = self.decoder1(z)
        h2 = self.decoder2(h1)
        return self.decoder3(h2)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return self.decode(z)