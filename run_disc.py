import numpy as np
import torch
import matplotlib.pyplot as plt
from pyDOE import lhs

from gpinn.network import FCN
from gpinn.training import TrainingPhase
from metrics import mse, rmse, mae
from residuals import pde_exponential_disc
from utils import phi_inter, load_data

Rd = 4
eta = 0.2
zd = Rd * eta
Md = 10 ** 10.5

# To generate new data:
z_min = 0
z_max = 4
r_min = 0
r_max = 80
total_points_z = 250
total_points_r = 250
# Nu: Number of training points # Nf: Number of collocation points (Evaluate PDE)
Nu = 296
Nf = 1024

z = torch.linspace(z_min, z_max, total_points_z).view(-1, 1) / zd
r = torch.linspace(r_min, r_max, total_points_r).view(-1, 1) / Rd
R, Z = torch.meshgrid(r.squeeze(1), z.squeeze(1), indexing='xy')

plt.figure()
plt.title('Potential interpolated')
plt.pcolormesh(R, Z, phi_inter(R, Z, scaled=False))
plt.xlabel("$R'$")
plt.ylabel("$z'$")
plt.colorbar(label="$\Phi(R, z)$")
plt.show()
plt.close()

left_X = torch.hstack((R[:, 0][:, None], Z[:, 0][:, None]))
left_Y = phi_inter(left_X[:, 0], Z[:, 0]).unsqueeze(1)

bottom_X = torch.hstack((R[0, :][:, None], Z[0, :][:, None]))
bottom_Y = phi_inter(bottom_X[:, 0], Z[0, :]).unsqueeze(1)

top_X = torch.hstack((R[-1, :][:, None], Z[-1, :][:, None]))
top_Y = phi_inter(top_X[:, 0], Z[-1, :]).unsqueeze(1)

right_X = torch.hstack((R[:, -1][:, None], Z[:, -1][:, None]))
right_Y = phi_inter(right_X[:, 0], Z[:, 0]).unsqueeze(1)

# VALIDATION DATA
z_val = torch.linspace(z_min+0.1, z_max-0.1, total_points_z//3).view(-1, 1)/zd
r_val = torch.linspace(r_min+0.1, r_max-0.1, total_points_r//3).view(-1, 1)/Rd
R_val, Z_val = torch.meshgrid(r_val.squeeze(1), z_val.squeeze(1), indexing='xy')

lb_val = torch.Tensor([r_val[0], z_val[0]])
ub_val = torch.Tensor([r_val[-1], z_val[-1]])

X_val = lb_val + (ub_val - lb_val) * lhs(2, Nf//3)
Y_val = phi_inter(X_val[:, 0], X_val[:, 1]).unsqueeze(1)

# TRAINING DATA
X_train = torch.vstack([left_X, bottom_X, top_X, right_X])
Y_train = torch.vstack([left_Y, bottom_Y, top_Y, right_Y])
# Choose(Nu) points of our available training data:
idx = np.random.choice(X_train.shape[0], Nu, replace=False)
X_train_Nu = X_train[idx, :]
Y_train_Nu = Y_train[idx, :]
# Collocation Points (Evaluate our PDe)
# Choose(Nf) points(Latin hypercube)
lb = torch.Tensor([r[0], z[0]])
ub = torch.Tensor([r[-1], z[-1]])
X_train_Nf = lb + (ub - lb) * lhs(2, Nf)  # 2 as the inputs are x and gamma

plt.figure()
plt.title("Configuration of training points")
plt.plot(X_train_Nu[:, 0], X_train_Nu[:, 1], 'xr', label="Boundary Points")
plt.plot(X_train_Nf[:, 0], X_train_Nf[:, 1], '.b', label="Collocation Points")
plt.plot(X_val[:, 0], X_val[:, 1], '.g', label="Validation points")
plt.xlabel('$R$')
plt.ylabel('$z$')
plt.legend(loc='upper right')
plt.show()
plt.close()

# Create Model
steps = 30_000
layers = np.array([2, 32, 32, 32, 32, 1])
PINN = FCN(layers)  # , act=torch.nn.SiLU())
print(PINN)
training = TrainingPhase(neural_net=PINN, training_points=(X_train_Nu, Y_train_Nu, X_train_Nf),
                         testing_points=(X_val, Y_val), equation=pde_exponential_disc, n_epochs=steps,
                         _loss_function=mse)

net, epochs, train_losses, val_losses, accuracies = training.train_model(optimizer=torch.optim.Adam, learning_rate=1e-5)

np.save("arrays/loss_disc.npy", train_losses)
np.save("arrays/epochs_disc.npy", epochs)
training.save_model("models/exp_disc.pt")

plt.figure()
plt.title('Training loss')
plt.plot(epochs, train_losses, label="Training Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.show()

plt.figure()
plt.title("Mean relative accuracy")
plt.plot(epochs, np.abs(accuracies), label="Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.yscale('log')
plt.show()


R_test, z_test, phi_test = load_data(scaled=True, reshape=True)
R_test /= Rd
z_test /= zd

input_matrix = np.array([R_test.flatten(), z_test.flatten()]).T
phi_pred = net(input_matrix).detach().numpy().reshape(250, 250)

diff = phi_test - phi_pred
vmin, vmax = np.percentile(diff, (5, 95))

plt.figure()
plt.title(f"Relative Error on testing data from simulation\n"
          f"Max relative error: {float(np.abs((diff / phi_test)).max()): %}\n"
          f"Average relative error: {float(np.abs((diff / phi_test)).mean()): %}")
plt.pcolormesh(R_test, z_test, np.abs(diff / phi_test))
plt.xlabel("$R'$")
plt.ylabel("$z'$")
plt.colorbar(label="$\dfrac{\Delta \Phi(R, z)}{\Phi(R, z)}$")
plt.show()
print(f"Max relative error: {float(np.abs(diff / phi_test).max()): %}")
print(f"Average relative error: {float(np.abs((diff / phi_test).mean())): %}")

n_points = 500
R_test = np.linspace(0, 80, n_points) / Rd
z_test = np.linspace(0, 4, n_points) / zd
R_test, z_test = np.meshgrid(R_test, z_test, indexing='ij')
phi_test = phi_inter(R_test, z_test)

input_matrix = np.array([R_test.flatten(), z_test.flatten()]).T
phi_pred = net(input_matrix).detach().numpy().reshape(n_points, n_points)

diff = phi_test - phi_pred

plt.figure()
plt.title(f"Relative Error on unseen testing data from simulation\n"
          f"Max relative error: {float(np.abs(diff / phi_test).max()): %}\n"
          f"Average relative error: {float(np.abs(diff / phi_test).mean()): %}")
plt.pcolormesh(R_test, z_test, np.abs(diff / phi_test))
plt.xlabel("$R$")
plt.ylabel("$z$")
plt.colorbar(label="$\dfrac{\Delta \Phi(R, z)}{\Phi(R, z)}$")
plt.show()

print(f"Max relative error: {float(np.abs(diff / phi_test).max()): %}")
print(f"Average relative error: {float(np.abs(diff / phi_test).mean()): %}")
