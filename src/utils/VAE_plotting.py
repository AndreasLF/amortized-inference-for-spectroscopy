from matplotlib import pyplot as plt
from numpy import np
import scipy.stats as stats
import math

def plot_reconstructions(mu, log_var, x, x_recons):
    """Plot the reconstructions of the autoencoder
    
    Args:
        x (np.array): The original data
        x_recons (np.array): The reconstructed data
    """

    plt.clf()

    # Make 10x3 subplots
    fig, axs = plt.subplots(10, 2, figsize=(15, 15))

    # Plot the original data with the reconstructions
    for i in range(10):
        axs[i, 0].plot(x[i])
        axs[i, 0].plot(x_recons[i])
        # axs[i, 0].set_title("Reconstruction")


    # plot normal distributions with mean mu and log variance log_var
    for i in range(10):
        
        # COnvert log variance to variance
        sigma = np.exp(log_var[i])
        x = np.linspace(mu - 5, mu + 5, 100)
        axs[i, 1].plot(x, stats.norm.pdf(x, mu[i][0], sigma[0]))
        axs[i, 1].plot(x, stats.norm.pdf(x, mu[i][0], sigma[0]))



        axs[i, 1].set_title("Distribution of latent space")

    plt.savefig("VAE_SERS_reconstructions.png")