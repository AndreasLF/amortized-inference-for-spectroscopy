import torch, torch.utils, torch.distributions; torch.manual_seed(0)
import torch.nn as nn
import os, sys, dill, wandb

curr_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_script_dir)
sys.path.append(parent_dir)

from src.models.VAE_pseudo_voigt import VAE, VAE_TwoParams, VAE_TwoParamsSigmoid
from src.generate_data2 import pseudoVoigtSimulatorTorch
from src.SERS_dataset import IterDataset
from src.trainers.VAE_trainer import VAE_trainer

parent_dir = os.path.dirname(curr_script_dir)
results_dir = os.path.join(parent_dir, 'results')
VAE_results_dir = os.path.join(results_dir, '3_VAE_voigt_decoder')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()
from src.trainers.VAE_trainer import VAE_trainer

latent_dims = 2
num_batches_per_epoch = 10
learning_rate = 1e-3
beta = 1
optimizer = "adam"
batch_size = 100

epochs = 10
ps = pseudoVoigtSimulatorTorch(500)
generator = ps.generator(1, peaks = torch.tensor([250]), gamma = torch.tensor([20]), eta = torch.tensor([0.5]), alpha = (0.5,10), sigma = 0.5)

dset_train = IterDataset(generator)
dset_test = IterDataset(generator)

# Load the SERS dataset
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, pin_memory=cuda)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, pin_memory=cuda)

autoencoder = VAE_TwoParams(next(iter(train_loader))[0][1].shape, latent_dims, batch_size, ["c", "alpha"]).to(device) 

#==============================================================================
# Train the model
#==============================================================================

autoencoder, train_loss = VAE_trainer(autoencoder, train_loader, 
                                        optimizer=optimizer, epochs=epochs, 
                                        num_iterations_per_epoch=num_batches_per_epoch,
                                        lr=learning_rate,beta=beta, label = ["c", "alpha"])

print(train_loss)