from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')
from src.generate_data2 import pseudoVoigtSimulatorTorch

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

def plot_loss(train_loss, x, recons, z, labels, label_name, y, generator_num, tmp_img="ae_tmp2.png", width_in_cm = None):

    z = z.detach().cpu().numpy()
    x = x.detach().cpu().numpy()
    recons = recons.detach().cpu().numpy()

    width = 2

    if len(labels) == 2:
        width = 3

    fig, axs = plt.subplots(3, width, figsize=(15, 12))
    if width_in_cm is not None:
        fig = change_fig_size(fig, width_in_cm)

    # plot the loss
    axs[0, 0].plot(train_loss)
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')

    # make latent space scatter plot with colorbar for labels
    lab = labels[0].detach().cpu().numpy()
    sc = axs[0, 1].scatter(z[:, 0], z[:, 1], c=lab, cmap='viridis')
    cbar = fig.colorbar(sc, ax=axs[0, 1])
    axs[0, 1].set_title('Latent space')
    # axis labels
    axs[0, 1].set_xlabel('$z_0$')
    axs[0, 1].set_ylabel('$z_1$')  
    # Give colorbar a rotated text 
    text_cbar = f'$\\{label_name[0]}$' if label_name[0] != 'c' else label_name[0]
    cbar.set_label(text_cbar, rotation=270-180)
    cbar.ax.yaxis.set_label_coords(-1, 0.5)

    if width == 3:
        # make latent space scatter plot with colorbar for labels
        lab = labels[1].detach().cpu().numpy()
        sc = axs[0, 2].scatter(z[:, 0], z[:, 1], c=lab, cmap='viridis')
        cbar = fig.colorbar(sc, ax=axs[0, 2])
        axs[0, 2].set_title('Latent space')
        # axis labels
        axs[0, 2].set_xlabel('$z_0$')
        axs[0, 2].set_ylabel('$z_1$')  
        # Give colorbar a rotated text 
        text_cbar = f'$\\{label_name[1]}$' if label_name[1] != 'c' else label_name[1]
        cbar.set_label(text_cbar, rotation=270-180)
        cbar.ax.yaxis.set_label_coords(-1, 0.5) 


    # Title after first row of plots 
    fig.suptitle(f'Autoencoder results (generator {generator_num})', fontsize=suptitle_size, fontweight=suptitle_fontweight)

    ps = pseudoVoigtSimulatorTorch(500)
   
    # rcparams colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # get y-lim 
    y_min, y_max = x[:width*2].min(), x[:width*2].max()

    for i in range(width*2):
        c_ = y[:,0][i]
        eta_ = y[:,2][i]
        gamma_ = y[:,1][i]
        alpha_ = y[:,3][i]
        vp = alpha_  * ps.pseudo_voigt(500, c_, gamma_, eta_, height_normalize=True, wavenumber_normalize=True)

        ax = axs[i//width+1, i%width]
        # plot the reconstructions
        ax.plot(x[i], label='x', alpha=0.5, color = colors[0])
        ax.plot(vp.flatten(), label='Pure Voigt', color = colors[0])
        ax.plot(recons[i], label='$\\hat{x}$', color = colors[2])
        ax.set_ylim(y_min, y_max)
        ax.set_title('Reconstruction')
        ax.set_xlabel('Wavenumber')
        ax.set_ylabel('Intensity')
        ax.legend()


    plt.tight_layout()
    return plt, fig, tmp_img
