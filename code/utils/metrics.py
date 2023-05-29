import torch
import numpy as np
import matplotlib.pyplot as plt


def mse(residual: torch.Tensor):
    return residual.pow(2).mean()


def rmse(residual: torch.Tensor):
    return torch.sqrt(residual.pow(2).mean())


def mae(array: torch.Tensor):
    return torch.abs(array).mean()


def absolute_error(y_pred, y_true, plot=False):
    result = np.abs(y_pred - y_true)
    if not plot:
        return result

    plt.figure()
    plt.title(r"Error difference $$\hat{\phi} - \phi$$")
    plt.plot(result)
    plt.show()
    return result


def relative_error(y_pred, y_true, plot=False):
    result = abs((y_pred - y_true) / y_true)
    if not plot:
        return result

    plt.figure()
    plt.title(r"Relative error $$\dfrac{\Delta \phi}{\phi}$$")
    plt.plot(result)
    plt.show()
    return result
