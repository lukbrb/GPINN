import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
# print(plt.style.available)

if len(sys.argv) > 1:                
    saveplot = sys.argv[-1]
print(saveplot)
"""
- Loss as a function of iteration
- Potential for 2 or 3 different values of gamma: NN prediction vs analytical profile
- A grid/scatter plot with s on the x-axis, gamma on the y-axis color coded by the relative (or absolute) difference between NN prediction and the analytic profile.
"""

def plot_loss(n_epoch, loss_values):
    plt.figure()
    plt.title('Loss as a function of iteration')
    plt.plot(n_epoch, loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def plot_prediction(s_domain, y_pred, y_true):
    plt.figure()
    plt.title('Dehnen potential')
    plt.plot(s_domain, y_pred, '--k', label='Predicted Value')
    plt.plot(s_domain, y_true, color='red', label='Analytical Value')
    plt.xlabel(r'$s$')
    plt.ylabel(r'$/Phi$')
    plt.legend()
    plt.show()



def plot_scatter_s_gamma(s, gamma, diff):
    # Création du graphique
    fig, ax = plt.subplots()

    # Définition des axes
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$\gamma$')

    # Tracé du scatter plot coloré par la différence relative entre les prédictions NN et le profil analytique
    scatter = ax.scatter(s, gamma, c=diff, cmap='coolwarm')

    # Ajout de la barre de couleur
    cbar = plt.colorbar(scatter)
    cbar.set_label('Différence relative')

    # Affichage du graphique
    plt.show()


# Création des données de test
s = np.random.rand(100)
gamma = np.random.rand(100)
diff = np.random.rand(100)

plot_scatter_s_gamma(s, gamma, diff)