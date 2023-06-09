import sys
# import os
# print(os.getcwd())
# sys.path.append('../GPINN')

import numpy as np
import matplotlib.pyplot as plt
from models.get_models import dehnen
from gpinn import potentials
from utils.metrics import relative_error

plt.style.use('seaborn-v0_8-darkgrid')
# print(plt.style.available)

saveplot = True
if len(sys.argv) > 1:
    saveplot = sys.argv[-1]

"""
- Loss as a function of iteration
- Potential for 2 or 3 different values of gamma: NN prediction vs analytical profile
- A grid/scatter plot with s on the x-axis, gamma on the y-axis color coded by the relative (or absolute) difference between NN prediction and the analytic profile.
"""


def plot_loss(n_epoch, loss_values, val_loss, scale=('linear', 'linear')):
    plt.figure()
    plt.title('Loss as a function of iteration')
    plt.plot(n_epoch, loss_values, label="Training Loss")
    #plt.plot(n_epoch, val_loss, label="Validation Loss")
    plt.xscale(scale[0])
    plt.yscale(scale[1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def plot_prediction(s_domain, y_pred, y_true, gamma, ax=None):
    if ax:
        ax.plot(s_domain, y_pred, '--k', label='Predicted Value')
        ax.plot(s_domain, y_true, color='red', label='Analytical Value')
        ax.set_xlabel(r'$s$')
        ax.set_ylabel(r'$\Phi$')
        return ax

    plt.figure()
    plt.title(f'Dehnen potential for $\gamma={gamma}$')
    plt.plot(s_domain, y_pred, '--k', label='Predicted Value')
    plt.plot(s_domain, y_true, color='red', label='Analytical Value')
    plt.xlabel(r'$s$')
    plt.ylabel(r'$\Phi$')
    plt.legend()
    plt.show()


def plot_scatter_s_gamma(s, gamma, diff, title="Relative Difference"):
    # Création du graphique
    fig, ax = plt.subplots()

    # Définition des axes
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$\gamma$')

    # Tracé du scatter plot coloré par la différence relative entre les prédictions NN et le profil analytique
    scatter = ax.scatter(s, gamma, c=diff, cmap='coolwarm')

    # Ajout de la barre de couleur
    cbar = plt.colorbar(scatter)
    cbar.set_label(title)

    # Affichage du graphique
    plt.show()


def plot_image_s_gamma(s, gamma, diff, title="Relative Difference", cmap='coolwarm'):
    # Création du graphique
    fig, ax = plt.subplots()
    # Définition des axes$
    ax.set_title(f"Average error: {diff.mean():%}")
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$\gamma$')
    # Tracé de l'image continue colorée par la différence relative entre les prédictions NN et le profil analytique
    image = ax.imshow(diff, extent=[s.min(), s.max(), gamma.min(), gamma.max()], cmap=cmap, origin='lower',
                      aspect='auto')

    # Ajout de la barre de couleur
    cbar = plt.colorbar(image)
    cbar.set_label(title)
    vmin, vmax = np.percentile(diff, (5, 95))
    image.set_clim(vmin=vmin, vmax=vmax)

    # Affichage du graphique
    plt.show()


if __name__ == "__main__":
    # -------- Plot the loss --------------
    loss = np.load("code/arrays/loss_dehnen.npy")
    val_loss = np.load("code/arrays/val_loss_dehnen.npy")
    epochs = np.load("code/arrays/epochs_dehnen.npy")
    plot_loss(epochs, loss, val_loss, scale=('linear', 'log'))

    # -------- Plot the prediction -----------
    n_subplots = 3
    fig, axs = plt.subplots(n_subplots, n_subplots, figsize=(12, 12))
    gamma = np.linspace(0, 2.99, n_subplots**2)
    s = np.linspace(1e-1, 10, 1000)
    for i, ax in enumerate(axs.flat):
        ax.set_title(f"$\gamma =$ {gamma[i]:.2f}")
        y_pred = dehnen(s, gamma=gamma[i]).detach()
        y_true = potentials.dehnen(s, gamma=gamma[i])
        plot_prediction(s, y_pred=y_pred, y_true=y_true, gamma=gamma[i], ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    plt.show()

    # -------- Plot the diff depending on gamma and s -----------
    s = np.linspace(1e-2, 10, 1000)
    gammas = np.linspace(0, 2.99, 100)
    abs_diffs = []

    rel_diffs = []
    dehnen_true = []
    dehnen_predicted = []
    for gamma in gammas:
        y_pred = dehnen(s, gamma=gamma).detach().numpy().flatten()
        y_true = potentials.dehnen(s, gamma=gamma)
        rel_diff = relative_error(y_pred=y_pred, y_true=y_true)

        dehnen_true.append(y_true)
        dehnen_predicted.append(y_pred)
        rel_diffs.append(rel_diff)

    rel_diffs = np.array(rel_diffs)
    dehnen_true = np.array(dehnen_true)
    dehnen_predicted = np.array(dehnen_predicted)

    plot_image_s_gamma(s, gammas, rel_diffs, title='Relative difference')
    # plot_image_s_gamma(s, gammas, dehnen_true, title='Analytical values of Dehnen profile', cmap='viridis')
    # plot_image_s_gamma(s, gammas, rel_diffs, title='Predicted values of Dehnen profile', cmap='viridis')
