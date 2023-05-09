import numpy as np
import torch
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.interpolate import RegularGridInterpolator

from gpinn.network import FCN
from gpinn.training import TrainingPhase

Rd = 4
eta = 0.2
zd = Rd * eta
Md = 10 ** 10.5


def pde_residual(nn, x_pde):
    _r, _z = x_pde[:, 0].unsqueeze(1), x_pde[:, 1].unsqueeze(1)
    x_pde.requires_grad = True
    f = nn(x_pde)
    # -------- Differentiation w.r.t. R ----------------
    f_rz = torch.autograd.grad(f, x_pde, torch.ones(x_pde.shape[0], 1), retain_graph=True, create_graph=True)[0]
    f_r = f_rz[:, 0].unsqueeze(1)
    f_r = f_r * _r
    f_rrz = torch.autograd.grad(f_r, x_pde, torch.ones(x_pde.shape[0], 1), retain_graph=True, create_graph=True)[0]
    f_rr = f_rrz[:, 0].unsqueeze(1)
    # -------- Differentiation w.r.t. z ----------------
    f_z = f_rz[:, 1].unsqueeze(1)
    f_zzr = torch.autograd.grad(f_z, x_pde, torch.ones(x_pde.shape[0], 1), retain_graph=True, create_graph=True)[0]
    f_zz = f_zzr[:, 1].unsqueeze(1)

    _lhs = 1 / _r * f_rr + 1 / eta ** 2 * f_zz
    rhs = torch.exp(-_r) * torch.cosh(_z) ** (-2)
    return _lhs - rhs


def mse(residual: torch.Tensor):
    return residual.pow(2).mean()


def rmse(residual: torch.Tensor):
    return torch.sqrt(residual.pow(2).mean())


def mae(array: torch.Tensor):
    return torch.abs(array).mean()


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


def load_data(reshape=True, scaled=False):
    data = np.loadtxt("notebooks/test_phi_grid.dat")
    R_test, z_test, phi_test = data.T
    if scaled:
        phi_test /= (10 ** 10.5 / zd)
    if reshape:
        return R_test.reshape(250, 250), z_test.reshape(250, 250), phi_test.reshape(250, 250)

    return R_test, z_test, phi_test


def phi_inter(R, z, scaled=True):
    Md = 1
    _r, _z, _phi = load_data(reshape=True)
    _r /= Rd
    _z /= zd
    _phi *= zd
    f = RegularGridInterpolator((np.ascontiguousarray(_r[:, 0]), np.ascontiguousarray(_z[0, :])),
                                np.ascontiguousarray(_phi))
    if scaled:
        Md = 10 ** 10.5
    return torch.Tensor(f((R, z)) / Md)


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

# Transform the mesh into a 2-column vector
# Transform the mesh into a 2-column vector
y_real = phi_inter(R, Z)
x_test = torch.hstack((R.transpose(1, 0).flatten()[:, None], Z.transpose(1, 0).flatten()[:, None]))
y_test = y_real.transpose(1, 0).flatten()[:, None]  # Colum major Flatten (so we transpose it)
# Domain bounds
lb = x_test[0]  # first value
ub = x_test[-1]  # last value

X_train = torch.vstack([left_X, bottom_X, top_X, right_X])
Y_train = torch.vstack([left_Y, bottom_Y, top_Y, right_Y])
# Choose(Nu) points of our available training data:
idx = np.random.choice(X_train.shape[0], Nu, replace=False)
X_train_Nu = X_train[idx, :]
Y_train_Nu = Y_train[idx, :]
# Collocation Points (Evaluate our PDe)
# Choose(Nf) points(Latin hypercube)
X_train_Nf = lb + (ub - lb) * lhs(2, Nf)  # 2 as the inputs are x and gamma
# X_train_Nf = torch.vstack((X_train_Nf, X_train_Nu))  # Add the training points to the collocation point

plt.figure()
plt.title("Configuration of training points")
plt.plot(X_train_Nu[:, 0], X_train_Nu[:, 1], 'xr', label="Boundary Points")
plt.plot(X_train_Nf[:, 0], X_train_Nf[:, 1], '.b', label="Collocation Points")
plt.xlabel('$R$')
plt.ylabel('$z$')
plt.legend(loc='upper right')
plt.show()
plt.close()

X_test = x_test.float()  # the input dataset (complete)
Y_test = y_test.float()  # the real solution

# Create Model
steps = 10_000
layers = np.array([2, 32, 32, 32, 1])

PINN = FCN(layers)  # , act=torch.nn.SiLU())

print(PINN)

training = TrainingPhase(neural_net=PINN, training_points=(X_train_Nu, Y_train_Nu, X_train_Nf),
                         testing_points=(X_test, Y_test), equation=pde_residual, n_epochs=steps,
                         _loss_function=mse)

net, epochs, losses = training.train_model(optimizer=torch.optim.Adam, learning_rate=1e-4)

np.save("arrays/loss_disc.npy", losses)
np.save("arrays/epochs_disc.npy", epochs)
training.save_model("models/exp_disc.pt")

plt.figure()
plt.title('Training loss')
plt.plot(epochs, losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
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
plt.title("Relative Error on unscaled testing data")
plt.pcolormesh(R_test, z_test, diff/phi_test)
plt.xlabel("$R'$")
plt.ylabel("$z'$")
plt.colorbar(label="$\dfrac{\Delta \Phi(R, z)}{\Phi(R, z)}$")
plt.show()

n_points = 500
R_test = np.linspace(0, 80, n_points)/Rd
z_test = np.linspace(0, 4, n_points)/zd
R_test, z_test = np.meshgrid(R_test, z_test, indexing='ij')
phi_test = phi_inter(R_test, z_test)

input_matrix = np.array([R_test.flatten(), z_test.flatten()]).T
phi_pred = net(input_matrix).detach().numpy().reshape(n_points, n_points)

diff = phi_test - phi_pred

plt.figure()
plt.title("Relative Error")
plt.pcolormesh(R_test, z_test, diff/phi_test)
plt.xlabel("$R$")
plt.ylabel("$z$")
plt.colorbar(label="$\dfrac{\Delta \Phi(R, z)}{\Phi(R, z)}$")
plt.show()

print(f"Max relative error: {float((diff/phi_test).max()): %}")
print(f"Average relative error: {float((diff/phi_test).mean()): %}")
