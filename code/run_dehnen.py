import numpy as np
import torch
import matplotlib.pyplot as plt
from pyDOE import lhs

from gpinn.network import FCN
from gpinn.training import TrainingPhase
from gpinn.potentials import dehnen
from gpinn.residuals import pde_dehnen
from utils.metrics import mae
plt.style.use('seaborn-v0_8-darkgrid')

# ========================= PARAMETERS =========================
steps = 10_000

layers = np.array([2, 32, 16, 1])

# To generate new data:
x_min = 1e-2
x_max = 10
gamma_min = 0
gamma_max = 2.99
total_points_x = 300
total_points_gamma = 10
# Nu: Number of training points # Nf: Number of collocation points (Evaluate PDE)
Nu = 10
Nf = 1024
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
X_train_Nu = X_train  # [idx, :]
Y_train_Nu = Y_train  # [idx, :]
# Collocation Points (Evaluate our PDe)


# Choose(Nf) points(Latin hypercube)
X_train_Nf = lb + (ub - lb) * lhs(2, Nf)  # 2 as the inputs are x and gamma
X_train_Nf = torch.vstack((X_train_Nf, X_train_Nu))  # Add the training points to the collocation point

# f_hat = torch.zeros(X_train_Nf.shape[0], 1)  # to minimize function
plt.figure()
plt.title("Configuration of training points")
plt.plot(X_train_Nu[:, 0], X_train_Nu[:, 1], 'xr', label="Boundary Points")
plt.plot(X_train_Nf[:, 0], X_train_Nf[:, 1], '.b', label="Collocation Points")
plt.xlabel('$s$')
plt.ylabel(r'$\gamma$')
plt.legend()
plt.show()
plt.close()

X_test = x_test.float()  # the input dataset (complete)
Y_test = y_test.float()  # the real solution

# Create Model
PINN = FCN(layers) #, act=torch.nn.SiLU())

print(PINN)

training = TrainingPhase(neural_net=PINN, training_points=(X_train_Nu, Y_train_Nu, X_train_Nf),
                         testing_points=(X_test, Y_test), equation=pde_dehnen, n_epochs=steps, _loss_function=mae)

net, epochs, losses, _, _ = training.train_model(optimizer=torch.optim.Adam, learning_rate=1e-3)
torch.save(net.state_dict(), 'test.pth')
np.save("code/arrays/loss_dehnen.npy", losses)
np.save("code/arrays/epochs_dehnen.npy", epochs)
training.save_model("code/models/dehnen.pt")
