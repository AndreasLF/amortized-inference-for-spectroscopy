import torch
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.cuda.is_available()


def plot_loss(train_losses, test_losses):
    """Plot the training and validation loss
    
    Args:
        epochs (int): The number of epochs
        train_losses (list): The training loss
        test_losses (list): The validation loss
    """

    plt.clf()
    
    plt.title("Error")
    # Set the x axis label of the current axis.
    plt.xlabel("Epoch")
    # Set the y axis label of the current axis.
    plt.ylabel("Error")

    n = len(train_losses)

    plt.plot(np.arange(n), train_losses, color="black")
    plt.plot(np.arange(n), test_losses, color="gray", linestyle="--")
    plt.legend(['Training error', 'Validation error'])

    return plt
      
def plot_reconstructions(autoencoder, data_loader, w_h_of_subplot=(2,2)):
    """Plot the reconstructions of the autoencoder
    
    Args:
        autoencoder (nn.Module): The autoencoder
        data_loader (DataLoader): The data loader
        w_h_of_subplot (tuple, optional): The width and height of the subplot. Defaults to (2,2).
    
    """

    # Reset plot plt
    plt.clf()

    w, h = w_h_of_subplot
    j = w*h
    plot_pos = 1
    K = 0
    # Plot reconstructions
    for i in range(j):
        x, y = next(iter(data_loader))

        x_hat = autoencoder(x.to(device))
        # print(len(x_hat))
        x_hat = x_hat.to('cpu').detach().numpy()
        x = x.to('cpu').detach().numpy()


        # if x has more than one dimension
        if x.shape[0] > 1:

            # Loop through rows in x
            for k in range(0, min(j,x.shape[0])):

                #Get first row 
                x_ = x[k]
                x_hat_ = x_hat[k]
                # flatten the data
                x_ = x_.flatten()
                x_hat_ = x_hat_.flatten()

                plt.subplot(w, h, plot_pos)

                # if isinstance(y, list):
                #     y = y[2]
                # peak_pos_rounded = str(list([np.round(i.tolist(),1) for i in y]))
                plt.plot(x_)
                plt.plot(x_hat_, alpha=0.7)
                # plt.title(f"Peak location: {peak_pos_rounded}")
                plt.xlabel("Wavenumber")
                plt.ylabel("Intensity (a.u.)")

                K += 1
                plot_pos += 1

                if K == j:
                    break
        else:
            # flatten the data
            x_ = x.flatten()
            x_hat_ = x_hat.flatten()

            plt.subplot(3, 2, i+1)

            if isinstance(y, list):
                y = y[2]
            peak_pos_rounded = str(list([np.round(i.tolist(),1) for i in y]))
            plt.plot(x_)
            plt.plot(x_hat_, alpha=0.7)
            plt.title(f"Peak location: {peak_pos_rounded}")
            plt.xlabel("Wavenumber")
            plt.ylabel("Intensity (a.u.)")

        if K == j:
            break
        # break
    # Make plot title
    plt.suptitle(f"Reconstruction of SERS spectra")
    # show legend   
    plt.legend(['Original', 'Reconstruction'])
    plt.tight_layout()

    return plt

def plot_latent_space(autoencoder, data_loader, points_to_plot=100):
    # Reset plot plt
    plt.clf()

    points = 0
    
    x, y = next(iter(data_loader))

    # zs = np.empty((x.shape[0], 0))

    zs = []
    ys =[]

    

    # ys = np.empty((1, 0))

    while points < points_to_plot:
        x, y = next(iter(data_loader))
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        y = y[2].to('cpu').detach().numpy()

        # loop through rows in z and y
        for i, z_ in enumerate(z):
            #  Add to rows
            zs.append(z_)
            ys.append(y[i])


            points += 1

    # transpose
    zs = np.array(zs)
    ys = np.array(ys)


    plt.scatter(zs[:,0], zs[:, 1], c=ys, cmap='Oranges')
    cbar = plt.colorbar()

    cbar.set_label('Location of peak (c)', rotation=270-180)
    # place cbar label on left side of colorbar
    cbar.ax.yaxis.set_label_coords(-1, 0.5)
    plt.title("Latent space")
    plt.xlabel("Latent variable 1")
    plt.ylabel("Latent variable 2")
    plt.tight_layout()
    return plt

