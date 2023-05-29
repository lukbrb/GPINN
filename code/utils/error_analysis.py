import matplotlib.pyplot as plt
from gpinn.potentials import hernquist


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
