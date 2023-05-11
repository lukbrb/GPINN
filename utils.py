import random
import string
import sqlite3
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

zd = 0.8
Rd = 4


def load_data(reshape=True, scaled=False):
    data = np.loadtxt("notebooks/test_phi_grid.dat")
    R_test, z_test, phi_test = data.T
    if scaled:
        phi_test /= (10 ** 10.5 / zd)
    if reshape:
        return R_test.reshape(250, 250), z_test.reshape(250, 250), phi_test.reshape(250, 250)

    return R_test, z_test, phi_test


def phi_inter(r, z, scaled=True):
    Md = 1
    _r, _z, _phi = load_data(reshape=True)
    _r /= Rd
    _z /= zd
    _phi *= zd
    f = RegularGridInterpolator((np.ascontiguousarray(_r[:, 0]), np.ascontiguousarray(_z[0, :])),
                                np.ascontiguousarray(_phi))
    if scaled:
        Md = 10 ** 10.5
    return torch.Tensor(f((r, z)) / Md)


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


def create_connection(database):
    conn = None
    try:
        conn = sqlite3.connect(database)
    except sqlite3.Error as e:
        print(e)
    return conn


def create_table(conn):
    sql_create_results_table = """CREATE TABLE IF NOT EXISTS expdisc (
                                    id INTEGER PRIMARY KEY,
                                    model_name TEXT NOT NULL UNIQUE,
                                    mean_accuracy REAL NOT NULL,
                                    min_accuracy REAL NOT NULL,
                                    max_accuracy REAL NOT NULL,
                                    median_accuracy REAL NOT NULL
                                );"""

    try:
        cursor = conn.cursor()
        cursor.execute(sql_create_results_table)
    except sqlite3.Error as e:
        print(e)


def insert_result(conn, result):
    sql = """INSERT INTO expdisc (model_name, mean_accuracy, min_accuracy, max_accuracy, median_accuracy) 
    VALUES(?, ?, ?, ?, ?);"""
    cursor = conn.cursor()
    cursor.execute(sql, result)
    conn.commit()


def check_model_exists(conn, model_name):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM expdisc WHERE model_name=?", (model_name,))
    return cursor.fetchone() is not None

# Renvoie "True" si existe déjà
