from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from src.generate_data2 import pseudoVoigtSimulatorTorch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
# import gaussian_filter
from scipy.ndimage import gaussian_filter
import matplotlib.gridspec as gridspec

suptitle_size = 12
# Normal font weight
suptitle_fontweight = 'normal'
# suptitle_fontweight = 'bold'

rcparam = {'axes.labelsize': 4,
            'font.size': 6,
            'legend.fontsize': 6,
            'axes.titlesize': 10,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            # marker size 
            'lines.markersize': 4
            }
plt.rcParams.update(**rcparam)

def change_fig_size(fig, width_in_cm):
    width, height = fig.get_size_inches()
    fig.get_size_inches()
    height_ratio = height / width

    width_in_cm = 14
    width_in_in = width_in_cm / 2.54
    fig.set_size_inches(width_in_in, width_in_in * height_ratio)
    return fig

def plt_latent_space_ellipses(z, mu, sigma, y, label_name, generator_num, sigma_factor=2, width_in_cm=None):    
    lab = y


    fig, axs = plt.subplots(1, len(label_name), figsize=(15, 7))
    if width_in_cm is not None:
        fig = change_fig_size(fig, width_in_cm)


    # if label_name is a list 
    for n, lab_name in enumerate(label_name):
        if len(label_name) == 1:
            ax = axs
        else:
            ax = axs[n]
        # make two plots next to each other

        sc = ax.scatter(z[:, 0], z[:, 1], c=lab[n], cmap='viridis')
        ll = f'$\\{lab_name}$' if lab_name != 'c' else lab_name
        cbar = fig.colorbar(sc, ax=ax, label=ll)

        # Set the font size of the colorbar label
        cbar.ax.tick_params(labelsize=10)        # Hide sc
        sc.set_visible(False)
        ax.set_title(f'Latent space, ${sigma_factor}\\sigma$ from $\\mu$')
        # axis labels
        ax.set_xlabel('$z_1$')
        ax.set_ylabel('$z_2$')  
        # Give colorbar a rotated text 
        # cbar.set_label(text_cbar, rotation=270-180)
        # cbar.ax.yaxis.set_label_coords(-1, 0.5)

        mu_sorted = mu[lab[n].flatten().argsort()]
        sigma_sorted = sigma[lab[n].flatten().argsort()]
        # lab = lab[lab.argsort()]
        # make colors for the scatter plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(mu)))

        # plot ellipses 
        for i in range(len(mu)):
            # get mean and std of the latent space
            mean = mu_sorted[i]
            std = sigma_sorted[i]
            # get the angle of the ellipse
            angle = np.arctan(std[1]/std[0])
            # get the width and height of the ellipse
            width_ellipse = sigma_factor*std[0]
            height_elipse = sigma_factor*std[1]
            # create the ellipse

            ellipse = matplotlib.patches.Ellipse(xy=mean, width=width_ellipse, height=height_elipse, angle=angle, alpha=0.5, color=colors[i])
            # add the ellipse to the plot
            ax.add_patch(ellipse)
        ax.scatter(mu_sorted[:, 0], mu_sorted[:, 1], c=colors, marker='x', label="$\\mu$")
        plt.legend(frameon=True)

    plt.suptitle(f'Latent space (generator {generator_num})', fontsize=suptitle_size, fontweight=suptitle_fontweight)
    plt.tight_layout()
    return plt
    

def plt_reconstructions(x, x_hat, x_hat_mu, y, n=3, width_in_cm=None):
    # get default colors    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    print()

    ps = pseudoVoigtSimulatorTorch(500)

    h = 2
    w = 3


    fig, axs = plt.subplots(h, w, figsize=(15,10))
    if width_in_cm is not None:
        fig = change_fig_size(fig, width_in_cm)

    y_min, y_max = x[:h*w].min(), x[:h*w].max()
    for i in range(h*w):
        ax = axs[i//w, i%w]
        ax.plot(x[i].flatten(), label="Spectrum", alpha=0.5, color=colors[0])

        c_ = y[:,0][i]
        eta_ = y[:,2][i]
        gamma_ = y[:,1][i]
        alpha_ = y[:,3][i]
        vp = alpha_  * ps.pseudo_voigt(500, c_, gamma_, eta_, height_normalize=True, wavenumber_normalize=True)

        ax.plot(vp.flatten(), label='Pure voigt', color = colors[0])

        ax.plot(x_hat_mu[i].flatten(), label="Reconstruction ($\\mu$)", alpha=1, color=colors[2])
        # axs[i//n, i%n].plot(x_hat[i].flatten(), label="Reconstruction ($z$)", alpha=0.8, color=colors[1], linestyle="--")
        # ax.set_title("Reconstruction of SERS spectra")
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Intensity (a.u.)")
        ax.legend(frameon=True)
        ax.set_ylim(y_min, y_max)
    # margin between suptitle and figure
    plt.subplots_adjust(top=2)
    plt.suptitle("Reconstructions of Raman spectra", fontsize=suptitle_size, fontweight = suptitle_fontweight)
    
    plt.tight_layout()
    # plt.show()
    return plt


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
    # fig.suptitle(f"Epoch {epoch} of {epochs}")

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


    # lab = labels[0].to('cpu').detach().numpy()
    

    # sc = axs[2,0].scatter(z[:, 0], z[:, 1], c=lab, cmap='viridis')
    # cbar = fig.colorbar(sc, ax=axs[2,0])
    # # Hide sc
    # sc.set_visible(False)
    # axs[2,0].set_title('Latent space, $2\\sigma$ from $\\mu$')
    # # axis labels
    # axs[2,0].set_xlabel('$z_0$')
    # axs[2,0].set_ylabel('$z_1$')  
    # # Give colorbar a rotated text 
    # text_cbar = f'$\\{label_name[0]}$' if label_name[0] != 'c' else label_name[0]
    # cbar.set_label(text_cbar, rotation=270-180)
    # cbar.ax.yaxis.set_label_coords(-1, 0.5)

    # mu_sorted = mu[lab.flatten().argsort()]
    # sigma_sorted = sigma[lab.flatten().argsort()]
    # # lab = lab[lab.argsort()]
    # # make colors for the scatter plot
    # colors = plt.cm.viridis(np.linspace(0, 1, len(lab)))

    # # plot ellipses 
    # for i in range(len(mu)):
    #     # get mean and std of the latent space
    #     mean = mu_sorted[i]
    #     std = sigma_sorted[i]
    #     # get the angle of the ellipse
    #     angle = np.arctan(std[1]/std[0])
    #     # get the width and height of the ellipse
    #     width_ellipse = 2*std[0]
    #     height_elipse = 2*std[1]
    #     # create the ellipse

    #     ellipse = matplotlib.patches.Ellipse(xy=mean, width=width_ellipse, height=height_elipse, angle=angle, alpha=0.5, color=colors[i])
    #     # add the ellipse to the plot
    #     axs[2,0].add_patch(ellipse)



    # colors = plt.cm.rainbow(np.linspace(0, 1, len(mu[0])))
    # for j in range(width):
    #     axs[1, j].plot(x[j].flatten(), label="Original")
    #     axs[1, j].plot(recons[j].flatten(), label="Reconstruction")
    #     axs[1, j].plot(recons_mu[j].flatten(), label="Reconstruction mu")
    #     axs[1, j].set_title("Reconstruction of SERS spectra")
    #     axs[1, j].set_xlabel("Wavenumber")
    #     axs[1, j].set_ylabel("Intensity (a.u.)")
    #     axs[1, j].legend()

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
    return plt, fig, tmp_img

    
def plot_losses(train_loss, generator_num, width_in_cm=None):
    plt.clf()
    # plot 5 loses 
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.5))
    if width_in_cm is not None:
        fig = change_fig_size(fig, width_in_cm)
    for i, key in enumerate(train_loss.keys()):
        axs[i].plot(train_loss[key])
        axs[i].set_title(key)
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Loss')

    plt.suptitle(f"Losses (generator {generator_num})", fontsize=suptitle_size, fontweight = suptitle_fontweight)
    # suptitle size
    plt.tight_layout()

    return plt

def plot_losses_3_2(train_loss, generator_num, width_in_cm=None):
    # plot 5 loses 
    plt.clf()
    gs = gridspec.GridSpec(2, 6, width_ratios=[1, 1, 1, 1, 1, 1], height_ratios=[1, 1])

    fig = plt.figure(figsize=(15, 11))
    if width_in_cm is not None:
        fig = change_fig_size(fig, width_in_cm)

    subtitles = {"loss": "$-\\mathcal{L}$ (Loss)", "elbo": "$\\mathcal{L}$ (ELBO)", "kl": "$D_{KL}\\left (q(z|x)\\parallel\\mathcal{N}(0,I) \\right)$", "logpx": "$\\log p(x|z)$", "MSE": "$MSE$"}

    axes = [gs[0, :2], gs[0, 2:4], gs[0, 4:6], gs[1, 1:3], gs[1, 3:5]]
    for i, key in enumerate(train_loss.keys()):
        ax = plt.subplot(axes[i])
        ax.plot(train_loss[key])
        ax.set_title(subtitles[key])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        

    plt.suptitle(f"Losses (generator {generator_num})", fontsize=suptitle_size, fontweight = suptitle_fontweight)



    plt.tight_layout()

    return plt


def plt_recons_with_dist(x, x_hat_mu, mu, sigma, y, generator_num, w=3, h=2, width_in_cm=None):
    """ Plot reconstructions with distribution
    
    Args:
        x (np.array): A batch of the data
        x_hat_mu (np.array): A batch of the reconstructions from mu
        mu (np.array): The mu values
        sigma (np.array): Sigma
        y (np.array): y values
        generator_num (int): The generator number corresponding to which dataset is used
        w (int, optional): Number of columns in plot grid. Defaults to 3.
        h (int, optional): Number of rows in plot grid. Defaults to 2.
        
    Returns:
        matplotlib.pyplot: The plot object
    """

    plt.clf()

    # get rcparams colors from matplotlibrc
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # w = w
    # h = 2
    ratio = h/w
    # y_lim = (x[:n*n].min(), x[:n*n].max())
    y_lim = (x[:w*h].min(), x[:w*h].max())

    ps = pseudoVoigtSimulatorTorch(500)
    ws = np.linspace(0,1,500)



    fig, axs = plt.subplots(h, w, figsize=(15,15*ratio))
    if width_in_cm is not None:
        fig = change_fig_size(fig, width_in_cm)

    if generator_num != 1:
        mm = mu[:,0][:w*h]
        ss = sigma[:,0][:w*h]
        # make linspace in 2 dimensions 
        lin = np.linspace(0, 1, 100)
        lin = np.repeat(lin[:, np.newaxis], w*h, axis=1)
        top_hist_y_max = stats.norm.pdf(lin, mm, ss).max()
        # print(top_hist_y_max)

    if generator_num != 2:
        if generator_num == 1:
            mm = mu[:,0][:w*h]
            ss = sigma[:,0][:w*h]
        else:
            mm = mu[:,1][:w*h]
            ss = sigma[:,1][:w*h]
            # make linspace in 2 dimensions 
        lin = np.linspace(*y_lim, 100)
        lin = np.repeat(lin[:, np.newaxis], w*h, axis=1)
        right_hist_y_max = stats.norm.pdf(lin, mm, ss).max()
        # print(stats.norm.pdf(lin, mm, ss).shape)
        # print(right_hist_y_max)

    for i in range(w*h):
        if h == 1 or w == 1:
            ax = axs[i]
        else:
            ax = axs[i//w, i%w]

        c_ = y[:,0][i]
        eta_ = y[:,2][i]
        gamma_ = y[:,1][i]
        alpha_ = y[:,3][i]
        vp = alpha_  * ps.pseudo_voigt(500, c_, gamma_, eta_, height_normalize=True, wavenumber_normalize=True)

        # plot x 
        ax.plot(ws, x[i], color = colors[0], alpha = 0.6, label="x")
        ax.plot(ws, vp.flatten(), color = colors[0], alpha = 1, label="pure voigt")
        # ax.plot(x_hat[i], color = colors[1], label="reconstruction ($z$)")
        ax.plot(ws, x_hat_mu[i], color = colors[2], label="reconstruction ($\mu$)")
        ax.legend(frameon=True)

        # create new axes on the right and on the top of the current axes.
        divider = make_axes_locatable(ax)
        ax.set_ylim(*y_lim)
        # axHistx = divider.append_axes("top", size=1.2, pad=0.1, sharex=ax)


        if generator_num != 2: 
            axHisty = divider.append_axes("right", size=0.6, pad=0.1, sharey=ax)

            if generator_num == 3:
                mu1 = mu[i][1]
                sigma1 = sigma[i][1]
                # z1 = z[i][1]
            else:
                mu1 = mu[i][0]
                sigma1 = sigma[i][0]
                # z1 = z[i][0]
            axHisty.axhline(mu1, linestyle='--', alpha=0.5, color=colors[2])

            

            # axHisty.axhline(z1, linestyle='--', alpha=0.5, color=colors[1])
            # axHisty.axhline(alpha_[0], linestyle='--', alpha=0.5, color=colors[0])


            ax.axhline(mu1, linestyle='--', alpha=0.5, color=colors[2])
            # ax.axhline(z1, linestyle='--', alpha=0.5, color=colors[1])
            # ax.axhline(alpha_[0], linestyle='--', alpha=0.5, color=colors[0])

            # Make horizontal line 
            # axHistx.axhline(0, linestyle='--', alpha=0.5)

            linsp = np.linspace(y_lim[0], y_lim[1], 100)
            lab = "$\\alpha$" 
            axHisty.plot(stats.norm.pdf(linsp, mu1, sigma1),linsp, color=colors[3], label = "$p(z|x)$")
            # axHisty.plot(0, z1, marker='o', alpha=0.5, label="$z$", color = colors[1])
            # plot mu with a cross marker
            axHisty.plot(0, mu1, marker='X', alpha=0.5, label="$\\mu$", color = colors[2])
            axHisty.legend()
            axHisty.legend(frameon=True)
            axHisty.set_xlim(0, right_hist_y_max)

        if generator_num != 1:
            axHisty = divider.append_axes("top", size=0.6, pad=0.1, sharex=ax)

            mu2 = mu[i][0]
            sigma2 = sigma[i][0]
            axHisty.axvline(mu2, linestyle='--', alpha=0.5, color=colors[2])

            ax.axvline(mu2, linestyle='--', alpha=0.5, color=colors[2])

            # Make horizontal line 
            # axHistx.axhline(0, linestyle='--', alpha=0.5)

            linsp = np.linspace(0, 1, 100)
            lab = "$\\alpha$" 
            axHisty.plot(linsp, stats.norm.pdf(linsp, mu2, sigma2), color=colors[3], label = "$p(z|x)$")
            # plot mu with a cross marker
            axHisty.plot(mu2, 0, marker='X', alpha=0.5, label="$\\mu$", color = colors[2])
            axHisty.legend(frameon=True)
            axHisty.set_ylim(0, top_hist_y_max)

        # axHisty.set_title("Latent space")

        ax.set_xlabel("$c$")
        ax.set_ylabel("$\\alpha$")

    # bold suptitle 
    # fig.suptitle("Reconstruction of $x$ with $z$ from $p(z|x)$", fontweight="bold")
    plt.suptitle(f"Reconstructions (generator {generator_num})", fontsize=suptitle_size, fontweight=suptitle_fontweight)



    plt.tight_layout()

    return plt

def plt_sigma_as_func_of_alpha_and_c(mu, sigma, y, width_in_cm=None):
    plt.clf()

    dict_ = {0: "c", 1: "\\alpha"}


    # subplot 2x2 
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    if width_in_cm is not None:
        fig = change_fig_size(fig, width_in_cm)

    for i in range(2):
        for j in range(2):
            ax = axs[i, j]

            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']

            # sort by sigma 
            # idx = np.argsort(y[:,3])
            if j == 0:
                idx = np.argsort(y[:,3].flatten())
            else:
                idx = np.argsort(y[:,0].flatten())
            y_ = y[idx]
            alpha_ = y_[:,3]
            c_ = y_[:,0]
            mu_ = mu[idx]
            # alpha_ = z_[:,1]    
            sigma_ = sigma[idx]

            # sigma_ = sigma_[:,i]
            if j == 0:
                ax.plot(alpha_, sigma_[:,i], alpha = 0.5, label = f"$\\sigma_{dict_[i]}$")
                ax.plot(alpha_, gaussian_filter(sigma_[:,i], sigma=5), label = f"$\\sigma_{dict_[i]}$ smoothed")
                ax.set_xlabel(f"$\\alpha$")
                ax.set_title(f"$\\sigma_{dict_[i]}$ as a function of $ \\alpha$")

            else:
                ax.plot(c_, sigma_[:,i], alpha = 0.5, label = f"$\\sigma_{dict_[i]}$")
                ax.plot(c_, gaussian_filter(sigma_[:,i], sigma=5), label = f"$\\sigma_{dict_[i]}$ smoothed")
                ax.set_xlabel(f"$c$")
                ax.set_title(f"$\\sigma_{dict_[i]}$ as a function of $c$")

            ax.set_ylabel(f"$\\sigma_{dict_[i]}$")
            ax.legend(frameon=True)

    return plt

def plot_kernels(autoencoder, layer_num=0, width_in_cm=None):

    kernels = autoencoder.encoder1[layer_num].weight.detach().cpu().numpy()
    num_kernels = kernels.shape[0]

    if num_kernels == 1:
        if width_in_cm is not None:
            fig = plt.figure()
            fig = change_fig_size(fig, width_in_cm)
        # plot the kernel
        plt.plot(kernels[0][0])
        plt.title(f"Convolutional layer kernel")
        plt.tight_layout()

        plt.show()

    else:
        # plot the kernels. max 4 plots in a row
        fig, axs = plt.subplots(num_kernels//4, 4, figsize=(15, 10))
        if width_in_cm is not None:
            fig = change_fig_size(fig, width_in_cm)
        for i in range(num_kernels):
            ax = axs[i//4, i%4]
            ax.plot(kernels[i][0])
            ax.set_title(f"Channel {i+1}")

        plt.suptitle(f"Convolutional layer kernels", fontsize=suptitle_size, fontweight=suptitle_fontweight)
        plt.tight_layout()

    return plt

def slide_kernel_over_signal(autoencoder, signal, layer_num=0, width_in_cm=None):
    kernels = autoencoder.encoder1[layer_num].weight.detach().cpu().numpy()
    num_kernels = kernels.shape[0]

    # plot the kernels. max 4 plots in a row
    fig, axs = plt.subplots(num_kernels//4, 4, figsize=(15, 10))
    if width_in_cm is not None:
        fig = change_fig_size(fig, width_in_cm)
    for i in range(num_kernels):
        ax = axs[i//4, i%4]
        # ax.plot(kernels[i][0])
        ax.plot(signal.flatten(), label="signal", alpha=0.5)
        ax.plot(np.correlate(kernels[i][0].flatten(), signal.flatten(), mode="valid"), label="convoluted signal")
        ax.set_title(f"Channel {i+1}")
        ax.legend(frameon=True)

    plt.suptitle(f"Cross correlation between kernels and signal")
    plt.tight_layout()
    return plt