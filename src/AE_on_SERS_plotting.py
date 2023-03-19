import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import dill
import os
import sys

# Get working directory, parent directoy, data and results directory
# Get working directory, parent directoy, data and results directory
cwd = os.getcwd()
curr_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_script_dir)
results_dir = os.path.join(parent_dir, 'results')
data_dir = os.path.join(parent_dir, 'data')

from SERS_dataset import SERSDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()

def plot_loss(epochs, train_losses, test_losses):
    """Plot the training and validation loss
    
    Args:
        epochs (int): The number of epochs
        train_losses (list): The training loss
        test_losses (list): The validation loss
    """
    
    plt.title("Error")
    # Set the x axis label of the current axis.
    plt.xlabel("Epoch")
    # Set the y axis label of the current axis.
    plt.ylabel("Error")

    plt.plot(np.arange(epochs), train_losses, color="black")
    plt.plot(np.arange(epochs), test_losses, color="gray", linestyle="--")
    plt.legend(['Training error', 'Validation error'])

    plt.show()

def plot_reconstructions(autoencoder, data_loader):
    # Plot reconstructions
    for i, (x, y) in enumerate(data_loader):
        x_hat = autoencoder(x.to(device))
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

def plot_latent_space(autoencoder, data_loader):

    # Loop through points in dataset and plot them
    for i, (x, y) in enumerate(data_loader):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()

        plt.scatter(z[:,0], z[:, 1], c=y, cmap='Oranges')
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
   

if __name__ == "__main__":
    # Load model file
    # dill_file = "SERS_autoencoder_2023-03-15_100epochs.dill"
    dill_file = "SERS_autoencoder_2023-03-19_200epochs_2latdims_SGD.dill"
    with open(f'{results_dir}\\{dill_file}', 'rb') as f:
        model_data = dill.load(f)

    # Unpack model data
    autoencoder = model_data['model']
    date = model_data['date']
    epochs = model_data['epochs']
    train_losses = model_data['train_loss']
    test_losses = model_data['test_loss']
    train_data = model_data['train_data']
    test_data = model_data['test_data']

    # Load the SERS dataset
    dset_train = SERSDataset(f"{data_dir}/SERS_data/{train_data}")
    dset_test = SERSDataset(f"{data_dir}/SERS_data/{test_data}")
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, pin_memory=cuda)
    test_loader = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, pin_memory=cuda)

    # Plot the training and validation loss
    plot_loss(epochs, train_losses, test_losses)
    plot_latent_space(autoencoder, test_loader)
    plot_reconstructions(autoencoder, test_loader)
