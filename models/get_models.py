""" Module for returning the trained models."""

from pathlib import Path

import numpy as np
import torch

from gpinn import network


def load_model(model_file: str) -> network.FCN:
    modele_path = Path.cwd() / model_file

    if modele_path.exists():
        model = torch.load(modele_path)
    else:
        raise ValueError(f"The model \'{model_file.split('.')[0]}\' does not exist or is not yet available.")

    return model


def hernquist(s, a=1):
    model = load_model("hernquist.pt")
    return model(s, a)


def dehnen(s, gamma, a=1):
    model = load_model("models/dehnen.pt")
    gamma_tab = np.ones_like(s) * gamma
    input_matrix = np.array([s, gamma_tab]).T
    return model(input_matrix)
