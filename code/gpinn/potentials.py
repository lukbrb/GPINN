""" Module gathering the gravitational potentials that are going to be used
    during the work :
    'Application of Physics-Informed Neural Network for Galaxy Dynamics'
    Note that we set here GM/a to unity.
    author: Lucas Barbier-Goy
    date: 2023-02-28
    reference: ...
"""

import numpy as np

# TODO: Ajouter annotation de type


def hernquist(radius, scale_factor=1):
    """ Value of the gravitational potential
            in the case of a Hernquist profile
    """
    return -scale_factor/(radius + scale_factor)


def dehnen(radius, gamma, scale_factor=1, tensor=True):
    """ Value of the gravitational potential
        in the case of a Dehnen profile
    """
    power1 = 2 - gamma
    if tensor:
        return -1 / power1 * (1 - (radius / (radius + scale_factor)) ** power1)
    else:
        if gamma == 2:
            return np.log(radius / (radius + scale_factor))

        return -1 / power1 * (1 - (radius / (radius + scale_factor)) ** power1)
