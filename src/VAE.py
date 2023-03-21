import math 
import torch
from torch import nn, Tensor
from matplotlib import pyplot as plt
from torch.distributions import Distribution
import numpy as np
from tqdm import tqdm
import os 
# Get working directory, parent directoy, data and results directory
cwd = os.getcwd()
curr_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_script_dir)
results_dir = os.path.join(parent_dir, 'results')
data_dir = os.path.join(parent_dir, 'data')
from SERS_dataset import SERSDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()

#==============================================================================
# VAE network class
#==============================================================================

class VanillaVAE(nn.Module):
    def __init__(self, in_features=500, latent_dim = 2):
        """ Vanilla VAE
        
        Args:
            in_features (int): The number of input features
            latent_dim (int): The number of latent dimensions
        """
        super(VanillaVAE, self).__init__()
        
        self.in_features = in_features

        # Define the encoder network
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
        )

        # Two linear layers to output mean and variance of latent Gaussian
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)

        # Define the decoder network
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.in_features)
            )


    def encode(self, input):
        """
        Encodes the input into the latent space.

        Args:
            input (Tensor): input to the encoder

        Returns:
            (Tensor): mean and log variance of the latent Gaussian distribution
        """
        result = self.encoder(input)
        # result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the latent codes to the original input space.

        Args:
            z (Tensor): latent codes

        Returns:
            (Tensor): reconstructed input
        """
        # result = self.decoder_input(z)
        result = self.decoder(z)
        # result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) 
        mu + sigma * epsilon
        epsilon ~ N(0, I)

        Args:
            mu (Tensor): mean of the latent Gaussian distribution
            logvar (Tensor): log variance of the latent Gaussian distribution
        
        Returns:
            (Tensor): sampled latent vector
        """
        # Calculate standard deviation from log variance
        sigma = torch.exp(0.5 * logvar)
        # Sample numbers from a standard normal distribution with same size as mu
        # eps = torch.randn_like(std)
        eps = torch.empty_like(mu).normal_()
        # Reparameterization trick
        return mu + sigma * eps

    def forward(self, input):
        """
        Forward pass of the VAE.

        Args:
            input (Tensor): input to the VAE

        Returns:
            (Tensor): reconstructed input, input, mu, log_var, and z       
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var, z]

    def loss_function(self, recons, input, mu, log_var, beta=1):
        """
        Computes the VAE loss function. The loss is composed of two terms:
        1. The reconstruction loss (the negative log probability of the input under the reconstructed Gaussian distribution
              induced by the decoder in the data space).
        2. The latent loss, which is defined as the Kullback-Leibler divergence between the distribution in latent space induced by the encoder on the data and some prior. 
                This acts as a kind of regularizer, and in this case we use a standard Gaussian prior.
    
        Args:
            recons (Tensor): the reconstructed input
            input (Tensor): the original input
            mu (Tensor): the latent mean
            log_var (Tensor): the latent log variance
            beta (float): weight of the latent loss term (default: 1)

        Returns:
            dict: containing the reconstruction loss, the latent loss, and the total loss
        """
        # MSE loss between input and reconstructed input
        recon_loss = ((input - recons)**2).sum() 

        # Kullback-Leibler divergence between the distribution in latent space induced by the encoder on the data and the Gaussian prior with mean 0 and variance 1
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # loss = recons_loss + beta * kld_loss
        loss = recon_loss + beta * kld_loss

        return {'recon_loss': recon_loss, 'loss': loss, 'KLD':-kld_loss.detach()}


#==============================================================================
# Train function
#==============================================================================

def train_vae(vae, train_loader, test_loader, optimizer, epochs=100, device="cpu", beta=1):
    for epoch in tqdm(range(epochs)):
        batch_loss = []
        valid_loss = []
        vae.train()
        for x, y in train_loader:
            x = x.to(device)

            # perform the forward pass
            recons, x, mu, log_var, z = vae(x)
            loss_dict = vae.loss_function(recons, x, mu, log_var, beta=beta) 
            loss = loss_dict['loss']
            batch_loss.append(loss.item())
            recon_loss = loss_dict['recon_loss']
            kld_loss = loss_dict['KLD']
            # {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(np.mean(batch_loss))

        with torch.no_grad():
            vae.eval()
            x, y = next(iter(test_loader))
            x = x.to(device)
            recons, x, mu, log_var, z = vae(x)

            loss_dict = vae.loss_function(recons, x, mu, log_var, beta=beta) 
            loss = loss_dict['loss']
            valid_loss.append(loss)
            recon_loss = loss_dict['recon_loss']
            kld_loss = loss_dict['KLD']

            valid_loss.append(loss.item())

        test_loss.append(np.mean(valid_loss))

            # print(loss.item())

    test_losses = np.array(test_loss)
    train_losses = np.array(train_loss)

    return vae, train_losses, test_losses

#==============================================================================
# Load the data
#==============================================================================
train_data = "1000_SERS_train_data_2023-03-15.csv"
test_data = "1000_SERS_test_data_2023-03-15.csv"
dset_train = SERSDataset(f"{data_dir}/SERS_data/{train_data}")
dset_test = SERSDataset(f"{data_dir}/SERS_data/{test_data}")

# Load the SERS dataset
batch_size = 100
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, pin_memory=cuda)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, pin_memory=cuda)


#==============================================================================
# Define the model
#==============================================================================
latent_features = 2
vae = VanillaVAE(in_features=500, latent_dim=latent_features)
print(vae)

#==============================================================================
# Train the model
#==============================================================================
beta = 1
# The Adam optimizer works really well with VAEs.
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

train_loss = []
test_loss = []

epochs = 100

train_vae(vae, train_loader, test_loader, optimizer, epochs=epochs, device=device, beta=beta)

# ==============================================================================
# Plot the results
# ==============================================================================

from AE_on_SERS_plotting import plot_loss


def plot_latent_space(vae, data_loader, device):
    """Plots the latent space of the VAE with standard normal distribution (2*sigma) and the distributions of the latents (2*sigma)
    
    Args:
        vae (VanillaVAE): The VAE model
        data_loader (torch.utils.data.DataLoader): The data loader
        device (torch.device): The device to use
    """
    # Golden ratio
    golden_ratio = (1 + 5**0.5) / 2
    width = 10

    fig, axes = plt.subplots(1, 1, figsize=(width, width/golden_ratio), squeeze=False)

    # Plot reconstructions
    for j, (x, y) in enumerate(data_loader):
        # Define the batch size
        batch_size = x.size(0)
        # Run the model
        x_hat, x, mu, logvar, z = vae(x.to(device))
        # Convert logvar to sigma
        sigma = torch.exp(0.5 * logvar)

        # Convert to numpy
        x_hat = x_hat.to('cpu').detach().numpy()
        z = z.to('cpu').detach().numpy()

        # Plot the latent space
        scale_factor = 2

        ax = axes[0, 0]
        plt.scatter(z[:,0], z[:,1], c=y, cmap='Oranges', s=[10]*batch_size)

        # Plot posterior for latent variables with scale_factor = 2
        for i in range(batch_size):
            mus = mu[i]
            sigmas = sigma[i]
            post = plt.matplotlib.patches.Ellipse((mus[0],mus[1]), scale_factor*sigmas[0], scale_factor*sigmas[1], color= plt.cm.Oranges(y[i]), fill=True, alpha=0.1)
            # Color the ellipse by its class
            post.set_facecolor(plt.cm.Oranges(y[i]))
            ax.add_artist(post)
        # # plot prior
        prior = plt.Circle((0, 0), scale_factor, color='gray', fill=True, alpha=0.1)
        ax.add_artist(prior)

    print(j)
    # add grid
    plt.grid()
    # make the plot square
    plt.axis('square')
    # remover borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.title("Latent space")
    plt.xlabel("Latent variable 1")
    plt.ylabel("Latent variable 2")
    # Colorbar legend
    cbar = plt.colorbar()

    cbar.set_label('Location of peak (c)', rotation=270-180)
    # place cbar label on left side of colorbar
    cbar.ax.yaxis.set_label_coords(-1, 0.5)
    plt.tight_layout()

    plt.show()   

def plot_VAE_reconstructions(vae, data_loader, device):
    """Plots the reconstructions of the VAE 

    Args:
        vae (VanillaVAE): The VAE model
        data_loader (torch.utils.data.DataLoader): The data loader
        device (torch.device): The device to use
    """ 
            
    # Plot reconstructions
    for i, (x, y) in enumerate(data_loader):
        print(x.shape, y.shape)
        x_hat, _, _, _, _ = vae(x.to(device))
        # print(len(x_hat))
        x_hat = x_hat.to('cpu').detach().numpy()
        x = x.to('cpu').detach().numpy()
        y = y.to('cpu').detach().numpy()

        for j in range(4):
            plt.subplot(2, 2, j+1)
            peak_pos_rounded = str(np.round(y[j],2))
            print(peak_pos_rounded)
            plt.plot(x[j])
            plt.plot(x_hat[j], alpha=0.7)
            plt.title(f"Peak location: {peak_pos_rounded}")
            plt.xlabel("Wavenumber")
            plt.ylabel("Intensity (a.u.)")
            
            # break
        # Make plot title
        plt.suptitle(f"Reconstruction of SERS spectra")
        # show legend   
        plt.legend(['Original', 'Reconstruction'])
        plt.tight_layout()
        plt.show()
        break

plot_loss(epochs, train_loss, test_loss)

plot_latent_space(vae, test_loader, device)
plot_VAE_reconstructions(vae, test_loader, device)
