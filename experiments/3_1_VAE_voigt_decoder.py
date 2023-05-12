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

batch_size = 100
latent_dims_list = [1]
epochs = 100
num_batches_per_epoch = 10
optimizer = "adam"
learning_rates = [0.001]
generators = {1: "alpha", 2: "c"}
betas = [0.1, 0.5, 1, 2, 5]

for generator_num, labels in generators.items():
    for learning_rate in learning_rates:
        for latent_dims in latent_dims_list:
            for beta in betas:
                #==============================================================================
                # Initialize logging
                #==============================================================================
                # start a new wandb run to track this script
                run = wandb.init(
                    # set the wandb project where this run will be logged
                    project="amortized-inference-for-spectroscopy",
                    
                    # track hyperparameters and run metadata
                    config={
                    "architecture": "VariationalAutoencoderVoigtDecoder",
                    "dataset": "generator_" + str(generator_num),
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "latent_space_dims": latent_dims,
                    "optimizer": optimizer,
                    "learning_rate": learning_rate,
                    "num_batches_per_epoch": num_batches_per_epoch,
                    "beta": beta
                    }
                )

                # Add a tag to identify the run
                run.tags = ["VAE_Voigt", "logmu"]

                #==============================================================================
                # Load the data
                #==============================================================================
                ps = pseudoVoigtSimulatorTorch(500)
                generator = ps.predefined_generator(generator_num)
                dset_train = IterDataset(generator)
                train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, pin_memory=cuda)

                #==============================================================================
                # Define the model
                #==============================================================================
                autoencoder = VAE(next(iter(train_loader))[0][1].shape, latent_dims, batch_size, labels).to(device) 

                #==============================================================================
                # Train the model
                #==============================================================================

                autoencoder, train_loss = VAE_trainer(autoencoder, train_loader, 
                                                        optimizer=optimizer, epochs=epochs, 
                                                        num_iterations_per_epoch=num_batches_per_epoch,
                                                        lr=learning_rate, beta=beta, label=labels, wandb_log=True)

                #==============================================================================
                # Save the model
                #==============================================================================
                to_save = {"model": autoencoder, "train_loss": train_loss, "generator": generator_num}
                # save the model
                with open(f'{VAE_results_dir}/{run.name}.dill', 'wb') as f:
                    dill.dump(to_save, f)

                # stop run
                run.finish()

                