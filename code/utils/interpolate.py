import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator

zd = 0.8
Rd = 4


def load_data(reshape=True, scaled=False):
    data = np.loadtxt("notebooks/test_phi_grid.dat")
    R_test, z_test, phi_test = data.T
    if scaled:
        phi_test /= (10 ** 10.5 / zd)
    if reshape:
        return R_test.reshape(250, 250), z_test.reshape(250, 250), phi_test.reshape(250, 250)

    return R_test, z_test, phi_test


def phi_inter(r, z, scaled=True):
    Md = 1
    _r, _z, _phi = load_data(reshape=True)
    _r /= Rd
    _z /= zd
    _phi *= zd
    f = RegularGridInterpolator((np.ascontiguousarray(_r[:, 0]), np.ascontiguousarray(_z[0, :])),
                                np.ascontiguousarray(_phi))
    if scaled:
        Md = 10 ** 10.5
    return torch.Tensor(f((r, z)) / Md)
