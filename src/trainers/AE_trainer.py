import torch 
import torch.nn as nn
import numpy as np
import os, sys

if os.path.basename(sys.argv[0]) == 'ipykernel_launcher.py':
    from IPython.display import Image, display, clear_output
    
from src.plotting.AE_plotting import plot_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()




def AE_trainer(autoencoder, data, optimizer="SGD", epochs=30, num_iterations_per_epoch = None, lr = 0.001, label = "alpha", wandb_log = False):
    """ Train the autoencoder on the data for a number of epochs
    
    Args:
        autoencoder (nn.Module): The autoencoder to train
        data (DataLoader): The data to train on
        epochs (int): The number of epochs to train for

    Returns:
        nn.Module: The trained autoencoder
    """
    if wandb_log:
        import wandb

    # The optimizer is defined 
    if optimizer == 'adam':
        opt = torch.optim.Adam(autoencoder.parameters(), lr = lr)
    else: 
        opt = torch.optim.SGD(autoencoder.parameters(), lr=lr)

    opt.zero_grad()

    # The loss function is defined
    loss_function = nn.MSELoss()

    train_loss = []

    # Loop through epochs 
    for epoch in range(epochs):
        batch_loss = []
        # Loop through batches of train data, one batch per epoch
        for i, (x, y) in enumerate(data):
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = loss_function(x_hat, x)
            loss.backward()
            opt.step()
            batch_loss.append(loss.item())
            if num_iterations_per_epoch and i == num_iterations_per_epoch:
                break
        
        train_loss.append(np.mean(batch_loss))

        if wandb_log:
            wandb.log({"train_loss": train_loss[-1]})

        # Only plot this if we are running in a notebook
        if os.path.basename(sys.argv[0]) == 'ipykernel_launcher.py':
            if epoch % 10 == 0:
                with torch.no_grad():
                    x, y = next(iter(data))
                    x = x.to(device)
                    z = autoencoder.encoder(x)
                    x_hat = autoencoder.decoder(z)

                    ll = {"c": 0, "gamma": 1, "eta": 2, "alpha": 3}
                    labels = []
                    if isinstance(label, list):
                        for l in label:
                            labels.append(y[:,ll[l]])
                    else:
                        labels.append(y[:,ll[label]])
                        label = [label]

                    plot, fig, tmp_img = plot_loss(train_loss, x, x_hat, z, labels, label, y)
                    
                    plot.savefig(tmp_img)
                    plot.close(fig)
                    display(Image(filename=tmp_img))
                    clear_output(wait=True)
                    os.remove(tmp_img)

    return autoencoder, train_loss