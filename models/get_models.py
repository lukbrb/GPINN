""" Module for returning the trained models."""

from pathlib import Path
import torch

from gpinn import network


def load_model(model_file: str) -> network.FCN:
    modele_path = Path.cwd() / model_file

    if modele_path.exists():
        model = torch.load(modele_path)
    else:
        raise ValueError(f"The model \'{model_file.split('.')[0]}\' does not exist or is not yet available.")

    return model


def hernquist():
    model = load_model("hernquist.pt")
    return model


def dehnen():
    model = load_model("dehnen.pt")
    return model
