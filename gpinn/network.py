import torch
from torch import nn


class FCN(nn.Module):
    def __init__(self, _layers, act=nn.Tanh()):
        super().__init__()
        self.layers = _layers
        self.activation = act
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i + 1])
                                      for i in range(len(self.layers) - 1)])
        self.iter = 0  # For the Optimizer

        for i in range(len(self.layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, _x):
        if not torch.is_tensor(_x):
            _x = torch.from_numpy(_x)
        a = _x.float()
        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
