import numpy as np
import torch
from pyDOE import lhs

from gpinn.network import FCN
from gpinn.training import TrainingPhase


# TODO: Changer la dérivée pour qu'elle ne soit qu'en fonction de x


def dehnen(radius, _gamma, scale_factor=1):
    """ Value of the gravitational potential
        in the case of a Dehnen profile
    """
    # if gamma == 2:
    # return np.log(radius / (radius + scale_factor))
    power1 = 2 - _gamma
    return (-1 / power1) * (1 - (radius / (radius + scale_factor)) ** power1)


def pde(nn, x_pde):
    _x, _gamma = x_pde[:, 0].unsqueeze(1), x_pde[:, 1].unsqueeze(1)
    x_pde.requires_grad = True  # Enable differentiation
    f = nn(x_pde)
    f_x = torch.autograd.grad(f, x_pde, torch.ones(x_pde.shape[0], 1), retain_graph=True, create_graph=True)[0]
    f_x = f_x[:, 0].unsqueeze(1)
    func = f_x * _x ** 2
    f_xx = torch.autograd.grad(func, x_pde, torch.ones(x_pde.shape[0], 1), retain_graph=True, create_graph=True)[0]
    return f_xx


# ========================= PARAMETERS =========================
steps = 50_000
lr = 1e-3
layers = np.array([2, 32, 32, 1])  # hidden layers
# To generate new data:
x_min = 1e-2
x_max = 10
gamma_min = 0
gamma_max = 2.99
total_points_x = 200
total_points_gamma = 100
# Nu: Number of training points # Nf: Number of collocation points (Evaluate PDE)
Nu = 100
Nf = 10000
# ========================= DATA GENERATION =========================

x = torch.linspace(x_min, x_max, total_points_x).view(-1, 1)
gamma = torch.linspace(gamma_min, gamma_max, total_points_gamma).view(-1, 1)
# Create the mesh
X, GAMMA = torch.meshgrid(x.squeeze(1), gamma.squeeze(1), indexing='xy')
y_real = dehnen(X, GAMMA)

# ========================= BOUNDARY CONDITIONS =========================
# First column # The [:,None] is to give it the right dimension
# ------------------------- Left Boundary -------------------------
left_X = torch.hstack((X[:, 0][:, None], GAMMA[:, 0][:, None]))
left_Y = dehnen(left_X[:, 0], GAMMA[:, 0]).unsqueeze(1)

# ------------------------- Lower Boundary -------------------------
bottom_X = torch.hstack((X[0, :][:, None], GAMMA[0, :][:, None]))
bottom_Y = dehnen(bottom_X[:, 0], GAMMA[0, :]).unsqueeze(1)

# ------------------------- Upper Boundary -------------------------
top_X = torch.hstack((X[-1, :][:, None], GAMMA[-1, :][:, None]))
top_Y = dehnen(top_X[:, 0], GAMMA[-1, :]).unsqueeze(1)

# ------------------------- Right Boundary -------------------------
right_X = torch.hstack((X[:, -1][:, None], GAMMA[:, -1][:, None]))
right_Y = dehnen(right_X[:, 0], GAMMA[:, 0]).unsqueeze(1)

# ------------------------- TESTING DATA -------------------------
# Transform the mesh into a 2-column vector
x_test = torch.hstack((X.transpose(1, 0).flatten()[:, None], GAMMA.transpose(1, 0).flatten()[:, None]))
y_test = y_real.transpose(1, 0).flatten()[:, None]  # Colum major Flatten (so we transpose it)
# Domain bounds
lb = x_test[0]  # first value
ub = x_test[-1]  # last value

# ------------------------- TRAINING DATA -------------------------
X_train = torch.vstack([left_X, bottom_X, top_X, right_X])
Y_train = torch.vstack([left_Y, bottom_Y, top_Y, right_Y])
# Choose(Nu) points of our available training data:
idx = np.random.choice(X_train.shape[0], Nu, replace=False)
X_train_Nu = X_train[idx, :]
Y_train_Nu = Y_train[idx, :]
# Collocation Points (Evaluate our PDe)


# TODO: Adapter ub  et lb à notre problème

# Choose(Nf) points(Latin hypercube)
X_train_Nf = lb + (ub - lb) * lhs(2, Nf)  # 2 as the inputs are x and t
X_train_Nf = torch.vstack((X_train_Nf, X_train_Nu))  # Add the training points to the collocation point

# f_hat = torch.zeros(X_train_Nf.shape[0], 1)  # to minimize function

X_test = x_test.float()  # the input dataset (complete)
Y_test = y_test.float()  # the real solution

# Create Model
PINN = FCN(layers)

print(PINN)

# optimizer = torch.optim.Adam(PINN.parameters(), lr=lr, amsgrad=False)

training = TrainingPhase(neural_net=PINN, training_points=(X_train_Nu, Y_train_Nu, X_train_Nf),
                         testing_points=(X_test, Y_test), equation=pde, n_epochs=steps,
                         optimizer=torch.optim.Adam,
                         _loss_function=torch.nn.MSELoss(reduction='mean'))

net, epochs, losses = training.train_model()
training.save_model("dehnen.pt")
