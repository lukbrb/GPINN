import matplotlib.pyplot as plt
import torch

from gpinn.potentials import hernquist


def erreur_difference(y_pred, y_true, plot=False):
    result = y_pred - y_true
    if not plot:
        return result

    plt.figure()
    plt.title(r"Error difference $$\hat{\phi} - \phi$$")
    plt.plot(result)
    plt.show()
    return result


def error_ratio(y_pred, y_true, plot=False):
    result = y_pred / y_true
    if not plot:
        return result

    plt.figure()
    plt.title(r"Error ratio $$\dfrac{\hat{\phi}}{\phi}$$")
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


def validation_loss(y_pred, y_true, _type="L2-norm"):
    absolute_error = abs(y_pred - y_true)
    if _type == "L2-norm":
        return absolute_error ** 2
    else:
        print("L2-norm not chosen. Using default L1-norm")
        return absolute_error


def check_testing_domain(model, domain, potential=(hernquist, 1), parameter=None, plot=False):
    if len(potential) == 2:
        potential_function, scale_factor = potential
        analytical_value = potential_function(domain, scale_factor)
    elif len(potential) == 3:
        potential_function, scale_factor, gamma = potential
        analytical_value = potential_function(domain, scale_factor, gamma)
    else:
        raise ValueError("Unvalid data specified for the potential.")
    if parameter:
        phi = model(domain, parameter)
    else:
        phi = model(domain)
    if not plot:
        return phi

    domain = domain.detach().numpy()
    phi = phi.detach().numpy()
    plt.figure()
    plt.title(f"{potential_function.__name__} potential")
    plt.plot(domain, phi, label="Predicted value")
    plt.plot(domain, analytical_value, label="Analytical value")
    plt.legend()
    plt.show()
    return phi


if __name__ == "__main__":
    net = torch.load("hernquist/resultats/Hernquist_50000.pt")
    x = torch.linspace(5, 50, 10000, requires_grad=False)
    x = x.reshape(-1, 1)
    prediction = net(x).detach().numpy()
    true_value = hernquist(x, 1).detach().numpy()
    loss = validation_loss(prediction, true_value)
    plt.figure()
    plt.plot(x, prediction, label="Predicted Value")
    plt.plot(x, true_value, label="Analytical Value")
    plt.legend()

    plt.figure()
    plt.plot(loss)
    plt.show()
    # check_testing_domain(net, x, plot=True)
