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

# Get working directory, parent directoy, data and results directory
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
results_dir = os.path.join(parent_dir, 'results')
data_dir = os.path.join(parent_dir, 'data')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()

def flatten(x):
    """Transforms a 28x28 image into a 784 vector
    From 02456 Deep Learning course https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch
    
    
    Args:
        x (Tensor): a 28x28 image

    Returns:
        Tensor: a 784 vector
    """
    tt = torchvision.transforms.ToTensor()
    return tt(x).view(28**2)

def stratified_sampler(labels, classes):
    """Sampler only picks datapoints in specified classes
    From 02456 Deep Learning course https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch
    
    Args:
        labels (Tensor): a tensor of labels
        classes (list): a list of classes to sample from
    
    Returns:
        SubsetRandomSampler: a sampler that only picks datapoints in specified classes
    
    """
    (indices,) = np.where(np.array([i in classes for i in labels]))
    indices = torch.from_numpy(indices)
    return torch.utils.data.sampler.SubsetRandomSampler(indices)

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
    

def train(autoencoder, data, epochs=30):
    """ Train the autoencoder on the data for a number of epochs
    
    Args:
        autoencoder (nn.Module): The autoencoder to train
        data (DataLoader): The data to train on
        epochs (int): The number of epochs to train for

    Returns:
        nn.Module: The trained autoencoder
    """

    # The optimizer is defined 
    # opt = torch.optim.Adam(autoencoder.parameters())
    opt = torch.optim.SGD(autoencoder.parameters(), lr=0.25)
    # The loss function is defined
    loss_function = nn.MSELoss()

    train_loss = []
    test_loss = []

    # Loop through epochs 
    for epoch in tqdm(range(epochs)):
        batch_loss = []
        valid_loss = []

        # Loop through batches of train data
        for x, y in data:
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = loss_function(x_hat, x)
            loss.backward()
            opt.step()

            batch_loss.append(loss.item())

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
# Load the MNIST data
dset_train = torchvision.datasets.MNIST(f"{data_dir}/MNIST/", train=True, download=True, transform=flatten)
dset_test  = torchvision.datasets.MNIST(f"{data_dir}/MNIST/", train=False, transform=flatten)

classes = [0,1]
batch_size = 64
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size,
                          sampler=stratified_sampler(dset_train.targets, classes), pin_memory=cuda)
test_loader  = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, 
                          sampler=stratified_sampler(dset_test.targets, classes), pin_memory=cuda)

# ==============================================================================
# Define the model
# ==============================================================================

latent_dims = 2

encoder = nn.Sequential(
            nn.Linear(in_features=28**2, out_features=128),
            nn.ReLU(),
            # bottleneck layer
            nn.Linear(in_features=128, out_features=latent_dims)
        ).to(device)

decoder = nn.Sequential(
            nn.Linear(in_features=latent_dims, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=28**2),
            nn.Sigmoid()
        ).to(device)

autoencoder = Autoencoder(encoder, decoder, latent_dims).to(device) 

print(autoencoder)

#==============================================================================
# Train the model
#==============================================================================

epochs = 5
autoencoder, train_loss, test_loss = train(autoencoder, train_loader, epochs=epochs)

today = date.today()

# Make dictionary with all the information
dictionary = {'model': autoencoder, 'train_loss': train_loss, 'test_loss': test_loss, 'epochs': epochs, 'date': today}

# save the model
with open(f'{results_dir}/MNIST_autoencoder_{today}_{epochs}epochs.dill', 'wb') as f:
    dill.dump(dictionary, f)
