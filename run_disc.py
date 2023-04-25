import numpy as np
import torch
import matplotlib.pyplot as plt
from pyDOE import lhs
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import RegularGridInterpolator

from gpinn.network import FCN
from gpinn.training import TrainingPhase


def pde_residual(nn, x_pde):
    eta = 0.2
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
    rhs = torch.exp(_r) * torch.cosh(_z) ** (-2)
    return _lhs - rhs


def mse(residual: torch.Tensor):
    return residual.pow(2).mean()


def rmse(residual: torch.Tensor):
    return torch.sqrt(residual.pow(2).mean())


def mae(array: torch.Tensor):
    return torch.abs(array).mean()


steps = 10_000

layers = np.array([2, 32, 16, 1])

# To generate new data:
z_min = 0
z_max = 4
r_min = 0
r_max = 80
total_points_z = 150
total_points_r = 150
# Nu: Number of training points # Nf: Number of collocation points (Evaluate PDE)
Nu = 50
Nf = 500


def minmax_scale(tensor: torch.Tensor):
    scaler = MinMaxScaler()
    scaler.fit(tensor.detach())
    return torch.Tensor(scaler.transform(tensor.detach()))


z = torch.linspace(z_min, z_max, total_points_z).view(-1, 1)
r = torch.linspace(r_min, r_max, total_points_r).view(-1, 1)
R, Z = torch.meshgrid(r.squeeze(1), z.squeeze(1), indexing='xy')


def load_data(reshape=True, scale=False):
    data = np.loadtxt("notebooks/test_phi_grid.dat")
    r_test, z_test, phi_test = data.T
    if scale:
        phi_test /= 10 ** 10.5
    if reshape:
        return r_test.reshape(250, 250), z_test.reshape(250, 250), phi_test.reshape(250, 250)

    return r_test, z_test, phi_test


def phi_inter(_r, _z, scale=True):
    md = 1
    _r, _z, _phi = load_data(reshape=True)
    f = RegularGridInterpolator((np.ascontiguousarray(_r[:, 0]), np.ascontiguousarray(_z[0, :])),
                                np.ascontiguousarray(_phi))
    if scale:
        md = 10 ** 10.5
    return torch.Tensor(f((_r, _z)) / md)


R_scaled = minmax_scale(R.T).T
Z_scaled = minmax_scale(Z)

left_X_scaled = torch.hstack((R_scaled[:, 0][:, None], Z_scaled[:, 0][:, None]))
left_Y_scaled = phi_inter(left_X_scaled[:, 0], Z_scaled[:, 0]).unsqueeze(1)

bottom_X_scaled = torch.hstack((R_scaled[0, :][:, None], Z_scaled[0, :][:, None]))
bottom_Y_scaled = phi_inter(bottom_X_scaled[:, 0], Z_scaled[0, :]).unsqueeze(1)

top_X_scaled = torch.hstack((R_scaled[-1, :][:, None], Z_scaled[-1, :][:, None]))
top_Y_scaled = phi_inter(top_X_scaled[:, 0], Z_scaled[-1, :]).unsqueeze(1)

right_X_scaled = torch.hstack((R_scaled[:, -1][:, None], Z_scaled[:, -1][:, None]))
right_Y_scaled = phi_inter(right_X_scaled[:, 0], Z_scaled[:, 0]).unsqueeze(1)

# Transform the mesh into a 2-column vector
y_real_scaled = phi_inter(R_scaled, Z_scaled)
x_test_scaled = torch.hstack((R_scaled.transpose(1, 0).flatten()[:, None], Z_scaled.transpose(1, 0).flatten()[:, None]))
y_test_scaled = y_real_scaled.transpose(1, 0).flatten()[:, None]  # Colum major Flatten (so we transpose it)
# Domain bounds
lb_scaled = x_test_scaled[0]  # first value
ub_scaled = x_test_scaled[-1]  # last value

X_train_scaled = torch.vstack([left_X_scaled, bottom_X_scaled, top_X_scaled, right_X_scaled])
Y_train_scaled = torch.vstack([left_Y_scaled, bottom_Y_scaled, top_Y_scaled, right_Y_scaled])
# Choose(Nu) points of our available training data:
idx = np.random.choice(X_train_scaled.shape[0], Nu, replace=False)
X_train_Nu_scaled = X_train_scaled[idx, :]
Y_train_Nu_scaled = Y_train_scaled[idx, :]
# Collocation Points (Evaluate our PDe)
# Choose(Nf) points(Latin hypercube)
X_train_Nf_scaled = lb_scaled + (ub_scaled - lb_scaled) * lhs(2, Nf)  # 2 as the inputs are x and gamma


plt.figure()
plt.title("Configuration of training points")
plt.plot(X_train_Nu_scaled[:, 0], X_train_Nu_scaled[:, 1], 'xr', label="Boundary Points")
plt.plot(X_train_Nf_scaled[:, 0], X_train_Nf_scaled[:, 1], '.b', label="Collocation Points")
plt.xlabel('$R$')
plt.ylabel('$z$')
plt.legend(loc='upper right')
plt.show()
plt.close()

X_test_scaled = x_test_scaled.float()  # the input dataset (complete)
Y_test_scaled = y_test_scaled.float()  # the real solution

# Create Model
PINN = FCN(layers)  # , act=torch.nn.SiLU())

print(PINN)

training = TrainingPhase(neural_net=PINN, training_points=(X_train_Nu_scaled, Y_train_Nu_scaled, X_train_Nf_scaled),
                         testing_points=(X_test_scaled, Y_test_scaled), equation=pde_residual, n_epochs=steps,
                         optimizer=torch.optim.Adam,
                         _loss_function=mse)

net, epochs, losses = training.train_model()

np.save("../arrays/loss_disc.npy", losses)
np.save("../arrays/epochs_disc.npy", epochs)
training.save_model("../models/exp_disc.pt")