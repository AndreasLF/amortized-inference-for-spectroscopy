import torch, torch.utils, torch.distributions; torch.manual_seed(0)
import torch.nn as nn
import os, sys, dill, wandb

curr_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_script_dir)
sys.path.append(parent_dir)

from src.AE_on_SERS import Autoencoder, train
from src.utils.AE_plotting import plot_loss, plot_reconstructions, plot_latent_space

parent_dir = os.path.dirname(curr_script_dir)
results_dir = os.path.join(parent_dir, 'results')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()


batch_sizes = [1, 10, 50, 100, 200]
latent_dims_list = [2, 3, 4, 5, 10, 20]
epochs = 1000
num_batches_per_epoch = 10
optimizer = "adam"
learning_rate = 0.001

for batch_size in batch_sizes:
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
            "architecture": "VanillaVAE",
            "dataset": "IterDataset",
            "tags": ["AutoencoderLinear"],
            "num_peaks": 1,
            "num_hotspots": 1,
            "batch_size": batch_size,
            "epochs": epochs,
            "latent_space_dims": latent_dims,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "num_batches_per_epoch": num_batches_per_epoch
            }
        )

        #==============================================================================
        # Load the data
        #==============================================================================

        from src.generate_data import SERS_generator_function
        from src.SERS_dataset import SERSDataset, IterDataset

        dset_train = IterDataset(SERS_generator_function(single_spectrum=True, num_peaks=1, num_hotspots=1))
        dset_test = IterDataset(SERS_generator_function(single_spectrum=True, num_peaks=1, num_hotspots=1))

        # Load the SERS dataset
        train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, pin_memory=cuda)
        test_loader = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, pin_memory=cuda)

        # ==============================================================================
        # Define the model
        # ==============================================================================

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

        #==============================================================================
        # Train the model
        #==============================================================================

        autoencoder, train_loss, test_loss = train(autoencoder, train_loader, test_loader, 
                                                optimizer=optimizer, epochs=epochs, 
                                                num_iterations_per_epoch=num_batches_per_epoch,
                                                lr=learning_rate,)


        # save the model
        with open(f'{results_dir}/wandb/{run.name}.dill', 'wb') as f:
            dill.dump(autoencoder, f)

        artifact = wandb.Artifact('model', type='dill')
        artifact.add_file(f'{results_dir}/wandb/{run.name}.dill')
        run.log_artifact(artifact)

        plot = plot_reconstructions(autoencoder, test_loader, (2,3))
        run.log({"reconstructions":wandb.Image(plot)})

        plot = plot_latent_space(autoencoder, test_loader, 200)
        run.log({"latent_space":wandb.Image(plot)})

        run.finish()