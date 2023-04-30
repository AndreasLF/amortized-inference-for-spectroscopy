import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.utils
import torch.distributions
import math
import numpy as np
from torch import Tensor
from torch.distributions import Distribution, Normal
from src.models.reparameterized_gaussian import ReparameterizedDiagonalGaussian
from src.generate_data2 import pseudoVoigtSimulatorTorch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()


class VAE(nn.Module):

    """ The autoencoder is a combination of the encoder and decoder
    
    Args:
        encoder (nn.Module): The encoder to use
        decoder (nn.Module): The decoder to use
        latent_dims (int): The number of latent dimensions to use
    """
    def __init__(self, input_shape, latent_dims, batch_size=100, decoder_type="alpha"):
        super(VAE, self).__init__()

        self.input_shape = input_shape
        self.latent_features = latent_dims
        self.observation_features = np.prod(input_shape)
        self.batch_size = batch_size

        self.encoder1 = nn.Linear(in_features=500, out_features=128).to(device)
        self.encoder2 = nn.ReLU().to(device)
        self.encoder_mu = nn.Linear(in_features=128, out_features=latent_dims).to(device)
        self.encoder_logvar = nn.Linear(in_features=128, out_features=latent_dims).to(device)

        # Pseudo-voigt decoder is used instead of a neural network. See decode function
        # self.decoder1 = nn.Linear(in_features=latent_dims, out_features=128).to(device)
        # self.decoder2 = nn.ReLU().to(device)
        # self.decoder3 = nn.Linear(in_features=128, out_features=500).to(device)

        self.kl = 0

        self.decoder_type = decoder_type

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_dims])))
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h1 = self.encoder1(x)
        h2 = self.encoder2(h1)
        mu = self.encoder_mu(h2)
        log_sigma = self.encoder_logvar(h2)
        
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
        return Normal(loc=px_logits, scale=0.25)

        # return Bernoulli(logits=px_logits, validate_args=False)

    def encode(self, x):
        h1 = self.encoder1(x)
        h2 = self.encoder2(h1)
        mu = self.encoder_mu(h2)
        logvar = self.encoder_logvar(h2)
        # torch rsample
        std = torch.exp(0.5*logvar)
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

        self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        
        # return {'px': px, 'pz': pz, 'qz': qz, 'z': z}

        return z, mu, logvar     
    

    def decode(self, z):
        # Decoder
        ps = pseudoVoigtSimulatorTorch(500)

        # peaks = torch.tensor([250])
        peaks = z if self.decoder_type == "c" else torch.tensor([250/500])
        gamma = torch.tensor([20]) 
        eta = torch.tensor([0.5])
        alpha = z if self.decoder_type == "alpha" else torch.tensor([5])
        pv = ps.decoder(peaks, gamma, eta, alpha, wavenumber_normalize=True, height_normalize=True, batch_size=self.batch_size)
        return pv

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return self.decode(z)
    
class VAE_TwoParams(VAE):
    #Inherit init from Autoencoder
    def __init__(self, input_shape, latent_dims, batch_size=100, decoder_type="alpha"):
        super(VAE_TwoParams, self).__init__(input_shape, latent_dims, batch_size, decoder_type)

    def decode(self, z):
        # Decoder
        ps = pseudoVoigtSimulatorTorch(500)

        # peaks = torch.tensor([250])
        peaks = z[:,0].reshape(-1,1)
        gamma = torch.tensor([20]) 
        eta = torch.tensor([0.5])
        alpha = z[:,1].reshape(-1,1)
        pv = ps.decoder(peaks, gamma, eta, alpha, wavenumber_normalize=True, height_normalize=True, batch_size=self.batch_size)
        return pv
        
class VAE_TwoParamsSigmoid(VAE_TwoParams):
    #Inherit init from Autoencoder
    def __init__(self, input_shape, latent_dims, batch_size=100, decoder_type="alpha"):
        super(VAE_TwoParamsSigmoid, self).__init__(input_shape, latent_dims, batch_size, decoder_type)


    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h1 = self.encoder1(x)
        h2 = self.encoder2(h1)
        mu = self.encoder_mu(h2)

        # sigmoid transform first column
        mu[:,0] = torch.sigmoid(mu[:,0])

        log_sigma = self.encoder_logvar(h2)
        
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
        return Normal(loc=px_logits, scale=0.25)

        # return Bernoulli(logits=px_logits, validate_args=False)

    def encode(self, x):
        h1 = self.encoder1(x)
        h2 = self.encoder2(h1)
        mu = self.encoder_mu(h2)
        # sigmoid transform first column
        mu[:,0] = torch.sigmoid(mu[:,0])
        logvar = self.encoder_logvar(h2)
        # torch rsample
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std

        
        x = x.view(x.size(0), -1)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        self.qz = self.posterior(x)
        
        # define the prior p(z)
        self.pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = self.qz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        self.px = self.observation_model(z)

        self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        
        # return {'px': px, 'pz': pz, 'qz': qz, 'z': z}

        return z, mu, logvar     
    

class VAE_TwoParamsSigmoidConv(VAE_TwoParamsSigmoid):
    """Variational Autoencoder with a two parameter pseudo voigt decoder for alpha and peak position. And a convolutional encoder """
    
    def __init__(self, input_shape, latent_dims, batch_size=100, decoder_type="alpha"):
        super(VAE_TwoParamsSigmoidConv, self).__init__(input_shape, latent_dims, batch_size, decoder_type)

        self.input_shape = input_shape
        self.latent_features = latent_dims
        self.observation_features = np.prod(input_shape)
        self.batch_size = batch_size

        # self.encoder1 = nn.Linear(in_features=500, out_features=128).to(device)
        # self.encoder2 = nn.ReLU().to(device)
        # self.encoder_mu = nn.Linear(in_features=128, out_features=latent_dims).to(device)
        # self.encoder_logvar = nn.Linear(in_features=128, out_features=latent_dims).to(device)


        # Make 1d convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=15, stride=1, padding=7)
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        # self.relu2 = nn.ReLU()
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        # self.relu3 = nn.ReLU()
        self.encoder_mu = nn.Linear(128 * 500, latent_dims)
        self.encoder_logvar = nn.Linear(128 * 500, latent_dims)
        # self.encoder_mu = nn.Linear(in_features=128, out_features=latent_dims).to(device)
        # self.encoder_logvar = nn.Linear(in_features=128, out_features=latent_dims).to(device)


        self.kl = 0

        self.decoder_type = decoder_type

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_dims])))
      
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""


        h = x.unsqueeze(1)  # Add a channel dimension for 1D convolution
        # print(h.shape)
        h1 = self.conv1(h)
        # print(h1.shape)
        h2 = self.relu1(h1)
        # print(h2.shape)
        # Make a linear layer 
        h2 = h2.view(h2.size(0), -1)
        # print(h2.shape)


        mu = self.encoder_mu(h2)
        mu[:,0] = torch.sigmoid(mu[:,0])

        log_sigma = self.encoder_logvar(h2)
        
        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def encode(self, x):


        h = x.unsqueeze(1)  # Add a channel dimension for 1D convolution
        # print(h.shape)
        h1 = self.conv1(h)
        h2 = self.relu1(h1)
        # print(h2.shape)
        # Make a linear layer 
        h2 = h2.view(h2.size(0), -1)
        # print(h2.shape)


        mu = self.encoder_mu(h2)
        mu[:,0] = torch.sigmoid(mu[:,0])

        logvar = self.encoder_logvar(h2)
        # print(logvar.shape)


        # h = x.unsqueeze(1)  # Add a channel dimension for 1D convolution
        # h1 = self.conv1(h)
        # h2 = self.relu1(h1)
        # h2 = h2.view(h2.size(0), -1)
        # # print(h2.shape)
        # mu = self.encoder_mu(h2)
        # # sigmoid transform first column
        # mu[:,0] = torch.sigmoid(mu[:,0])
        # logvar = self.encoder_logvar(h2)
        # torch rsample
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std

        
        x = x.view(x.size(0), -1)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        self.qz = self.posterior(x)
        
        # define the prior p(z)
        self.pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = self.qz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        self.px = self.observation_model(z)

        self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        
        # return {'px': px, 'pz': pz, 'qz': qz, 'z': z}

        return z, mu, logvar 
