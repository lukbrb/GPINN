import torch


def pde_dehnen(nn, x_pde):
    _x, _gamma = x_pde[:, 0].unsqueeze(1), x_pde[:, 1].unsqueeze(1)
    x_pde.requires_grad = True  # Enable differentiation
    f = nn(x_pde)
    f_x = torch.autograd.grad(f, x_pde, torch.ones(x_pde.shape[0], 1),
                              retain_graph=True, create_graph=True)[0]

    f_x = f_x[:, 0].unsqueeze(1)
    func = f_x * _x ** 2
    f_xx = torch.autograd.grad(func, x_pde, torch.ones(x_pde.shape[0], 1),
                               retain_graph=True, create_graph=True)[0]
    _gamma = x_pde[:, 1].unsqueeze(1)
    y_true = (2 * x_pde[:, 0].unsqueeze(1) ** (2 - _gamma)) / (x_pde[:, 0].unsqueeze(1) + 1) ** (4 - _gamma)
    return f_xx - y_true


def pde_exponential_disc(nn, x_pde):
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
    rhs = torch.exp(-_r) * torch.cosh(_z) ** (-2)
    return _lhs - rhs


