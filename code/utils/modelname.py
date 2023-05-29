import random
import string
import numpy as np


def random_string(length):
    # Créez un ensemble de caractères possibles (lettres majuscules, minuscules et chiffres)
    characters = string.ascii_letters + string.digits
    # Utilisez la fonction choice pour sélectionner un caractère aléatoire pour chaque position dans la chaîne
    random_str = ''.join(random.choice(characters) for _ in range(length))
    return random_str


def generate_model_name(activation, num_layers, num_neurons, error_func, learning_rate):
    activation_name = activation.__class__.__name__.lower()
    error_func_name = error_func.__name__.lower()
    learning_rate_exp = int(np.log10(learning_rate))

    model_name = f"{activation_name}{num_layers}{num_neurons}{error_func_name}1e{learning_rate_exp}"
    return model_name

