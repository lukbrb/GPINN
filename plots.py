import sys
import numpy as np
import matplotlib.pyplot as plt
from models.get_models import dehnen
from gpinn import potentials
from error_analysis import relative_error, erreur_difference
from sklearn.metrics import mean_squared_error
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


def plot_loss(n_epoch, loss_values, scale=('linear', 'linear')):
    plt.figure()
    plt.title('Loss as a function of iteration')
    plt.plot(n_epoch, loss_values)
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


def plot_image_s_gamma(s, gamma, diff, title="Relative Difference"):
    # Création du graphique
    fig, ax = plt.subplots()
    # Définition des axes
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$\gamma$')
    # Tracé de l'image continue colorée par la différence relative entre les prédictions NN et le profil analytique
    image = ax.imshow(diff, extent=[s.min(), s.max(), gamma.min(), gamma.max()], cmap='coolwarm', origin='lower',
                      aspect='auto')

    # Ajout de la barre de couleur
    cbar = plt.colorbar(image)
    cbar.set_label(title)

    # Affichage du graphique
    plt.show()


if __name__ == "__main__":
    # -------- Plot the loss --------------
    loss = np.load("arrays/loss.npy")
    epochs = np.load("arrays/epochs.npy")
    plot_loss(epochs, loss, scale=('linear', 'log'))

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
    gammas = np.linspace(0, 2, 100)
    abs_diffs = []
    rel_diffs = []
    for gamma in gammas:
        y_pred = dehnen(s, gamma=gamma).detach().numpy().flatten()
        y_true = potentials.dehnen(s, gamma=gamma)
        abs_diff = erreur_difference(y_pred=y_pred, y_true=y_true)
        rel_diff = relative_error(y_pred=y_pred, y_true=y_true)
        print(mean_squared_error(y_true=y_true, y_pred=y_pred))
        abs_diffs.append(abs_diff)
        rel_diffs.append(rel_diff)

    abs_diffs = np.array(abs_diffs)
    rel_diffs = np.array(rel_diffs)
    plot_image_s_gamma(s, gammas, abs_diffs, title='Absolute difference')
    plot_image_s_gamma(s, gammas, rel_diffs, title='Relative difference')
