import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import dill
from tqdm import tqdm
from datetime import date
import os
import sys
import datetime as dt
from matplotlib import pyplot as plt


# Get working directory, parent directoy, data and results directory
cwd = os.getcwd()
curr_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_script_dir)
results_dir = os.path.join(parent_dir, 'results')
data_dir = os.path.join(parent_dir, 'data')


from SERS_dataset import SERSDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()

class Autoencoder(nn.Module):
    """ The autoencoder is a combination of the encoder and decoder
    
    Args:
        encoder (nn.Module): The encoder to use
        decoder (nn.Module): The decoder to use
        latent_dims (int): The number of latent dimensions to use
    """
    def __init__(self, encoder, decoder, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
def train(autoencoder, data, optimizer="SGD", epochs=30, num_iterations_per_epoch = 0):
    """ Train the autoencoder on the data for a number of epochs
    
    Args:
        autoencoder (nn.Module): The autoencoder to train
        data (DataLoader): The data to train on
        epochs (int): The number of epochs to train for

    Returns:
        nn.Module: The trained autoencoder
    """

    # The optimizer is defined 
    if optimizer == 'adam':
        opt = torch.optim.Adam(autoencoder.parameters())
    else: 
        opt = torch.optim.SGD(autoencoder.parameters(), lr=10)

    # The loss function is defined
    loss_function = nn.MSELoss()
    

    train_loss = []
    test_loss = []

    # Loop through epochs 
    for epoch in tqdm(range(epochs)):
        batch_loss = []
        valid_loss = []

        # Loop through batches of train data
        for i, (x, y) in enumerate(data):
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = loss_function(x_hat, x)
            loss.backward()
            opt.step()

            batch_loss.append(loss.item())
            if type(train_loader.dataset) == IterDataset and i == num_iterations_per_epoch:
                break
        train_loss.append(np.mean(batch_loss))

        with torch.no_grad():
            autoencoder.eval()
            x, y = next(iter(test_loader))
            x = x.to(device)
            x_hat = autoencoder(x)

            loss = loss_function(x_hat, x)

            valid_loss.append(loss.item())

        test_loss.append(np.mean(valid_loss))
        
    return autoencoder, train_loss, test_loss

#==============================================================================
# Load the data
#==============================================================================
# train_data = "1000_SERS_train_data_2023-03-25.csv"
# test_data = "1000_SERS_test_data_2023-03-25.csv"
# dset_train = SERSDataset(f"{data_dir}/SERS_data/{train_data}")
# dset_test = SERSDataset(f"{data_dir}/SERS_data/{test_data}")

from generate_data import SERS_generator_function
from SERS_dataset import SERSDataset, IterDataset

dset_train = IterDataset(SERS_generator_function(single_spectrum=True, num_peaks=1, num_hotspots=1))
dset_test = IterDataset(SERS_generator_function(single_spectrum=True, num_peaks=1, num_hotspots=1))

# Load the SERS dataset
batch_size = 100
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, pin_memory=cuda)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, pin_memory=cuda)

# ==============================================================================
# Define the model
# ==============================================================================

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

autoencoder = Autoencoder(encoder, decoder, latent_dims).to(device) 

print(autoencoder)

#==============================================================================
# Train the model
#==============================================================================

epochs = 100
optimizer = 'adam'
autoencoder, train_loss, test_loss = train(autoencoder, train_loader, optimizer=optimizer, epochs=epochs)

today = date.today()
time_now = dt.datetime.now()

# Make dictionary with all the information
dictionary = {'model': autoencoder, 'train_loss': train_loss, 'test_loss': test_loss, 
              'epochs': epochs, 'optimizer': optimizer, "latent_space_dims": latent_dims, 
              'date': today, "train_data": train_data, "test_data": test_data, "time": time_now}

# save the model
with open(f'{results_dir}/SERS_autoencoder_{today}_{epochs}epochs_{latent_dims}latdims_{optimizer}.dill', 'wb') as f:
    dill.dump(dictionary, f)

