import torch; torch.manual_seed(0)
import torch.utils
import torch.distributions
import torch.nn as nn
import numpy as np
from torch import Tensor
import os, sys
if os.path.basename(sys.argv[0]) == 'ipykernel_launcher.py':
    from IPython.display import Image, display, clear_output

from src.plotting.VAE_plotting import plot_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()

def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

def VAE_trainer(autoencoder, data, optimizer="SGD", epochs=30, num_iterations_per_epoch = None, lr = 0.001, beta=1, label = "alpha"):
    """ Train the autoencoder on the data for a number of epochs
    
    Args:
        autoencoder (nn.Module): The autoencoder to train
        data (DataLoader): The data to train on
        epochs (int): The number of epochs to train for

    Returns:
        nn.Module: The trained autoencoder
    """

    MSE_loss = nn.MSELoss()

    # The optimizer is defined 
    if optimizer == 'adam':
        opt = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    else: 
        opt = torch.optim.SGD(autoencoder.parameters(), lr=lr)

    opt.zero_grad()

    # Lists to store the loss values
    train_loss = []
    train_loss_kl = []
    train_loss_elbo = []
    train_loss_logpx = []
    train_loss_MSE = []

    # Loop through epochs 
    for epoch in range(epochs):
        batch_loss = []
        batch_loss_kl = []
        batch_elbo = []
        batch_logpx = []
        batch_MSE = []
      
        # Loop through batches of train data
        for i, (x, y) in enumerate(data):
            x = x.to(device)
            opt.zero_grad()
            z, mu, logvar = autoencoder.encode(x)
            x_hat = autoencoder.decode(z)

            px = autoencoder.px
            pz = autoencoder.pz
            qz = autoencoder.qz

            # evaluate log probabilities
            log_px = reduce(px.log_prob(x))
            log_pz = reduce(pz.log_prob(z))
            log_qz = reduce(qz.log_prob(z))
            
            kl = log_qz - log_pz
            elbo = log_px - kl
            beta_elbo = log_px - beta * kl
            
            # loss
            loss = -beta_elbo.mean()
        
            loss.backward()
            opt.step()

            batch_loss.append(loss.mean().item())
            batch_loss_kl.append(kl.mean().item())
            batch_elbo.append(elbo.mean().item())
            batch_logpx.append(log_px.mean().item())
            batch_MSE.append(MSE_loss(x_hat, x).mean().item())
            
            if num_iterations_per_epoch and i == num_iterations_per_epoch:
                break

        train_loss.append(np.mean(batch_loss))
        train_loss_kl.append(np.mean(batch_loss_kl))
        train_loss_elbo.append(np.mean(batch_elbo))
        train_loss_logpx.append(np.mean(batch_logpx))
        train_loss_MSE.append(np.mean(batch_MSE))
         
        # if it is a notebook, show the plots
        if os.path.basename(sys.argv[0]) == 'ipykernel_launcher.py':
            with torch.no_grad():
                autoencoder.eval()
                x, y = next(iter(data))
                x = x.to(device)
                z, mu, logvar = autoencoder.encode(x)
                x_hat = autoencoder.decode(z)
                x_hat_mu = autoencoder.decode(mu)
            
                ll = {"c": 0, "gamma": 1, "eta": 2, "alpha": 3}
                labels = []
                if isinstance(label, list):
                    for l in label:
                        labels.append(y[:,ll[l]])
                else:
                    labels.append(y[:,ll[label]])
                    label = [label]

            if epoch % 10 == 0:
                plot, fig, tmp_img = plot_loss(epoch+1, epochs, train_loss, train_loss_kl, train_loss_elbo, train_loss_logpx,  z, x, x_hat, x_hat_mu, mu, logvar, labels, label)
                plot.savefig(tmp_img)
                plot.close(fig)
                display(Image(filename=tmp_img))
                clear_output(wait=True)

                os.remove(tmp_img)

    train_loss = {"loss": train_loss, "kl": train_loss_kl, "elbo": train_loss_elbo, "logpx": train_loss_logpx, "MSE": train_loss_MSE}
    return autoencoder, train_loss