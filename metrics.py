import torch


def mse(residual: torch.Tensor):
    return residual.pow(2).mean()


def rmse(residual: torch.Tensor):
    return torch.sqrt(residual.pow(2).mean())


def mae(array: torch.Tensor):
    return torch.abs(array).mean()
