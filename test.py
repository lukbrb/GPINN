import numpy as np
import torch
from gpinn.network import FCN

layers = np.array([1, 32, 1])
net_cpu = FCN(layers)
net_mps = FCN(layers)

net_mps.to(torch.device("mps"))


def pde(nn, _device):
    x_pde = torch.linspace(1e-2, 10, 100, device=_device).view(-1, 1)
    torch.autograd.set_detect_anomaly(True)
    x_pde.requires_grad = True  # Enable differentiation
    f = nn(x_pde)
    f_x = torch.autograd.grad(f, x_pde, torch.ones_like(x_pde).to(_device),
                              retain_graph=True, create_graph=True)[0]

    f_x = f_x[:, 0].unsqueeze(1)
    f_x *= x_pde ** 2
    # f_xx = torch.autograd.grad(f_x, x_pde, torch.ones_like(x_pde).to(_device),
    #                          retain_graph=True, create_graph=True)[0]
    return f_x


x = 0.1
cpu_tensor = torch.exp(torch.tensor(x))
mps_tensor = torch.exp(torch.tensor(x, device=torch.device("mps")))
print(cpu_tensor - cpu_tensor)  # prints 0
print(mps_tensor - mps_tensor)  # prints 0
print(cpu_tensor - mps_tensor)  # prints 1.1921e-07
print(cpu_tensor - mps_tensor.cpu())  # prints 1.1921e-07
print(cpu_tensor.to(torch.device("mps")) - mps_tensor)  # prints 1.1921e-07
cpu = pde(net_cpu, torch.device("cpu"))
mps = pde(net_mps, torch.device("mps"))
# print(mps)
# print((cpu - mps.cpu()))
