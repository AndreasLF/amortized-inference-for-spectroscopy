import torch, torch.utils, torch.distributions; torch.manual_seed(0)
import torch.nn as nn
import os, sys, dill, wandb

curr_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_script_dir)
sys.path.append(parent_dir)

from src.models.autoencoder import Autoencoder
from src.generate_data2 import pseudoVoigtSimulatorTorch
from src.SERS_dataset import IterDataset
from src.trainers.AE_trainer import AE_trainer

parent_dir = os.path.dirname(curr_script_dir)
results_dir = os.path.join(parent_dir, 'results')
AE_results_dir = os.path.join(results_dir, '1_AE')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()

batch_size = 100
latent_dims_list = [2]
epochs = 10000
num_batches_per_epoch = 10
optimizer = "adam"
learning_rates = [0.01]
generators = {1: "alpha", 2: "c", 3: ["c", "alpha"]}
# generators = {3: ["c", "alpha"]}

for generator_num, labels in generators.items():
    for learning_rate in learning_rates:
        for latent_dims in latent_dims_list:
            #==============================================================================
            # Initialize logging
            #==============================================================================
            # start a new wandb run to track this script
            run = wandb.init(
                # set the wandb project where this run will be logged
                project="amortized-inference-for-spectroscopy",
                
                # track hyperparameters and run metadata
                config={
                "architecture": "Autoencoder",
                "dataset": "generator_" + str(generator_num),
                "batch_size": batch_size,
                "epochs": "until convergence",
                "latent_space_dims": latent_dims,
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "num_batches_per_epoch": num_batches_per_epoch
                }
            )

            # Add a tag to identify the run
            run.tags = ["AE"]

            ps = pseudoVoigtSimulatorTorch(500)
            generator = ps.predefined_generator(generator_num)
            dset_train = IterDataset(generator)
            train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, pin_memory=cuda)

            # Load the SERS dataset
            autoencoder = Autoencoder(latent_dims).to(device) 

            autoencoder, train_loss = AE_trainer(autoencoder, train_loader, 
                                                    optimizer=optimizer, epochs=epochs, 
                                                    num_iterations_per_epoch=num_batches_per_epoch,
                                                    lr=learning_rate, label=labels, wandb_log=True)


            to_save = {"model": autoencoder, "train_loss": train_loss, "generator": generator_num}
            # save the model
            with open(f'{AE_results_dir}/{run.name}.dill', 'wb') as f:
                dill.dump(to_save, f)

            # stop run
            run.finish()
