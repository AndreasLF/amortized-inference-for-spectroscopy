from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from src.models.reparameterized_gaussian import ReparameterizedDiagonalGaussian
import os, sys
from IPython.display import Image, display, clear_output

def plt_latent_space_ellipses(z, mu, sigma, y, label_name):
    lab = y

    sc = plt.scatter(z[:, 0], z[:, 1], c=lab, cmap='viridis')
    cbar = plt.colorbar(label=label_name, orientation="vertical")
    # Hide sc
    sc.set_visible(False)
    plt.title('Latent space, $2\\sigma$ from $\\mu$')
    # axis labels
    plt.xlabel('$z_0$')
    plt.ylabel('$z_1$')  
    # Give colorbar a rotated text 
    # cbar.set_label(text_cbar, rotation=270-180)
    # cbar.ax.yaxis.set_label_coords(-1, 0.5)

    mu_sorted = mu[lab.flatten().argsort()]
    sigma_sorted = sigma[lab.flatten().argsort()]
    # lab = lab[lab.argsort()]
    # make colors for the scatter plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(lab)))

    # plot ellipses 
    for i in range(len(mu)):
        # get mean and std of the latent space
        mean = mu_sorted[i]
        std = sigma_sorted[i]
        # get the angle of the ellipse
        angle = np.arctan(std[1]/std[0])
        # get the width and height of the ellipse
        width_ellipse = 2*std[0]
        height_elipse = 2*std[1]
        # create the ellipse

        ellipse = matplotlib.patches.Ellipse(xy=mean, width=width_ellipse, height=height_elipse, angle=angle, alpha=0.5, color=colors[i])
        # add the ellipse to the plot
        plt.gca().add_patch(ellipse)
    

def plt_reconstructions(x, x_hat, x_hat_mu, y, full_spec, n=3):
    # get default colors    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    print()

    fig, axs = plt.subplots(n, n, figsize=(15,15))
    for i in range(n*n):
        axs[i//n, i%n].plot(x[i].flatten(), label="Spectrum", alpha=0.5, color=colors[0])
        if  y.shape[1] == 1:
            axs[i//n, i%n].plot(full_spec(y[i]).flatten(), label="Pure voigt", color=colors[0])
        else:
            axs[i//n, i%n].plot(full_spec(y[:,0][i],y[:,3][i]).flatten(), label="Pure voigt", color=colors[0])
        axs[i//n, i%n].plot(x_hat_mu[i].flatten(), label="Reconstruction ($\\mu$)", alpha=0.7, color=colors[2])
        axs[i//n, i%n].plot(x_hat[i].flatten(), label="Reconstruction ($z$)", alpha=0.8, color=colors[1], linestyle="--")
        axs[i//n, i%n].set_title("Reconstruction of SERS spectra")
        axs[i//n, i%n].set_xlabel("Wavenumber")
        axs[i//n, i%n].set_ylabel("Intensity (a.u.)")
        axs[i//n, i%n].legend()
        axs[i//n, i%n].legend(frameon=True)

    plt.tight_layout()
    plt.show()

def plot_loss(epoch, epochs, loss, loss_kl, loss_elbo, loss_logpx,  z, x, recons, recons_mu, mu, logvar, labels, label_name, tmp_img="plots_vae_temp2.png"):
    """ Plot the loss over time
    
    Args:
        loss (list): The loss over time

    Returns:
        None
    """
    x = x.to('cpu').detach().numpy()
    recons = recons.to('cpu').detach().numpy()
    recons_mu = recons_mu.to('cpu').detach().numpy()
    mu = mu.to('cpu').detach().numpy()
    logvar = logvar.to('cpu').detach().numpy()
    sigma = np.exp(0.5*logvar)
    z = z.to('cpu').detach().numpy()
    
    width = 4
    fig, axs = plt.subplots(3, width, figsize=(15, 15))

    # Figure title above all subplots
    fig.suptitle(f"Epoch {epoch} of {epochs}")

    # Plot loss
    axs[0, 0].plot(loss, label="Total loss")
    axs[0, 0].set_xlabel("Iteration")
    axs[0, 0].set_title("Train loss")

    axs[0, 1].plot(loss_kl, label = "beta KL loss")
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_title("KL")

    axs[0, 2].plot(loss_elbo, label = "ELBO")
    axs[0, 2].set_xlabel("Iteration")
    axs[0, 2].set_title("ELBO")

    axs[0, 3].plot(loss_logpx, label = "log-likelihood")
    axs[0, 3].set_xlabel("Iteration")
    axs[0, 3].set_title("Log-likelihood")

    
    # if z.shape[1] == 1: 
    #     colors = plt.cm.rainbow(np.linspace(0, 1, z.shape[0]))
    #     # randomize color order
    #     np.random.shuffle(colors)
    #     # Make spacing between points on y-axis
    #     y = np.linspace(0, 1, z.shape[0])
    #     # make horizontal lines 2*sigma above and below the mean
    #     axs[0,3].hlines(y, mu[:,0]-2*sigma[:,0], mu[:,0]+2*sigma[:,0], color=colors, alpha=0.4)
    #     axs[0,3].scatter(z[:,0], y, color=colors, s=10)
    #     # make y-axis taller
    #     # remove ticks on y axis
    #     axs[0,3].set_yticks([])
    #     # SET LABEL FOR X AXIS
    #     text_cbar = f'$\\{label_name[0]}$' if label_name[0] != 'c' else label_name[0]

    #     axs[0,3].set_xlabel(text_cbar)

    # else:
        # colors = plt.cm.rainbow(np.linspace(0, 1, len(mu)))
        # axs[0,3].plot(z[:,0], z[:,1], 'o', alpha=0.5, c=colors)
        # # get color from color map 
        # # scatter plot with different colors
        # # axs[0,2].scatter(z[:, 0], z[:, 1], c=colors, s = 3, alpha=0.9)

        # # plot ellipses for each mu and sigma
        # for i in range(len(mu)):
        #     mu_x = mu[i][0]
        #     mu_y = mu[i][1]
        #     sigma_x = sigma[i][0]
        #     sigma_y = sigma[i][1]
        #     # plot ellipse
        #     axs[0,3].scatter(mu_x, mu_y, color = colors[i], s=1, alpha=0.5)
        #     ellipse = matplotlib.patches.Ellipse((mu_x, mu_y), 2*sigma_x, 2*sigma_y, alpha=0.08, color=colors[i])
        #     axs[0,3].add_patch(ellipse)

        # axs[0,3].set_xlabel("z1")
        # axs[0,3].set_ylabel("z2")
        # axs[0,3].set_title("Latent space")


    lab = labels[0].to('cpu').detach().numpy()
    

    sc = axs[2,0].scatter(z[:, 0], z[:, 1], c=lab, cmap='viridis')
    cbar = fig.colorbar(sc, ax=axs[2,0])
    # Hide sc
    sc.set_visible(False)
    axs[2,0].set_title('Latent space, $2\\sigma$ from $\\mu$')
    # axis labels
    axs[2,0].set_xlabel('$z_0$')
    axs[2,0].set_ylabel('$z_1$')  
    # Give colorbar a rotated text 
    text_cbar = f'$\\{label_name[0]}$' if label_name[0] != 'c' else label_name[0]
    cbar.set_label(text_cbar, rotation=270-180)
    cbar.ax.yaxis.set_label_coords(-1, 0.5)

    mu_sorted = mu[lab.flatten().argsort()]
    sigma_sorted = sigma[lab.flatten().argsort()]
    # lab = lab[lab.argsort()]
    # make colors for the scatter plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(lab)))

    # plot ellipses 
    for i in range(len(mu)):
        # get mean and std of the latent space
        mean = mu_sorted[i]
        std = sigma_sorted[i]
        # get the angle of the ellipse
        angle = np.arctan(std[1]/std[0])
        # get the width and height of the ellipse
        width_ellipse = 2*std[0]
        height_elipse = 2*std[1]
        # create the ellipse

        ellipse = matplotlib.patches.Ellipse(xy=mean, width=width_ellipse, height=height_elipse, angle=angle, alpha=0.5, color=colors[i])
        # add the ellipse to the plot
        axs[2,0].add_patch(ellipse)



    colors = plt.cm.rainbow(np.linspace(0, 1, len(mu[0])))
    for j in range(width):
        axs[1, j].plot(x[j].flatten(), label="Original")
        axs[1, j].plot(recons[j].flatten(), label="Reconstruction")
        axs[1, j].plot(recons_mu[j].flatten(), label="Reconstruction mu")
        axs[1, j].set_title("Reconstruction of SERS spectra")
        axs[1, j].set_xlabel("Wavenumber")
        axs[1, j].set_ylabel("Intensity (a.u.)")
        axs[1, j].legend()

        # for i in range(len(mu[0]-1)):
        #     axs[2, j].axvline(mu[j][i], linestyle='--', alpha=0.5, color=colors[i])
        #     linsp = np.linspace(np.min(mu[j])-sigma[j][np.argmin(mu[j])]*3, np.max(mu[j])+sigma[j][np.argmax(mu[j])]*3, 100)
        #     print(label_name)
        #     lab = f"$\\{label_name[i]}$" if label_name[i] != 'c' else label_name[i] 
        #     axs[2, j].plot(linsp, stats.norm.pdf(linsp, mu[j][i], sigma[j][i]), color=colors[i], label=lab)
        #     axs[2, j].plot(z[j][i], 0, 'o', color=colors[i], alpha=0.5)
        #     axs[2, j].set_title("Latent space")
        #     axs[2, j].legend()
            # axs[2, j].set_xlabel("Wavenumber")
        

    plt.tight_layout()
    plt.savefig(tmp_img)
    plt.close(fig)
    display(Image(filename=tmp_img))
    clear_output(wait=True)

    os.remove(tmp_img)
