import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch import nn
from pyDOE import lhs

from gpinn.network import FCN
from gpinn.training import TrainingPhase
from utils.metrics import mse
from gpinn.residuals import pde_exponential_disc
from utils import (phi_inter, create_connection, create_table,
                   insert_result, check_model_exists, generate_model_name)


DATABASE = True
Rd = 4
eta = 0.2
zd = Rd * eta
Md = 10 ** 10.5

# To generate new data:
z_min = 0
z_max = 4
r_min = 0
r_max = 80
total_points_z = 250
total_points_r = 250
# Nu: Number of training points # Nf: Number of collocation points (Evaluate PDE)
Nu = 296
Nf = 1024

z = torch.linspace(z_min, z_max, total_points_z).view(-1, 1) / zd
r = torch.linspace(r_min, r_max, total_points_r).view(-1, 1) / Rd
R, Z = torch.meshgrid(r.squeeze(1), z.squeeze(1), indexing='xy')

left_X = torch.hstack((R[:, 0][:, None], Z[:, 0][:, None]))
left_Y = phi_inter(left_X[:, 0], Z[:, 0]).unsqueeze(1)

bottom_X = torch.hstack((R[0, :][:, None], Z[0, :][:, None]))
bottom_Y = phi_inter(bottom_X[:, 0], Z[0, :]).unsqueeze(1)

top_X = torch.hstack((R[-1, :][:, None], Z[-1, :][:, None]))
top_Y = phi_inter(top_X[:, 0], Z[-1, :]).unsqueeze(1)

right_X = torch.hstack((R[:, -1][:, None], Z[:, -1][:, None]))
right_Y = phi_inter(right_X[:, 0], Z[:, 0]).unsqueeze(1)

# VALIDATION DATA
z_val = torch.linspace(z_min + 0.1, z_max - 0.1, total_points_z // 3).view(-1, 1) / zd
r_val = torch.linspace(r_min + 0.1, r_max - 0.1, total_points_r // 3).view(-1, 1) / Rd
R_val, Z_val = torch.meshgrid(r_val.squeeze(1), z_val.squeeze(1), indexing='xy')

lb_val = torch.Tensor([r_val[0], z_val[0]])
ub_val = torch.Tensor([r_val[-1], z_val[-1]])

X_val = lb_val + (ub_val - lb_val) * lhs(2, Nf // 3)
Y_val = phi_inter(X_val[:, 0], X_val[:, 1]).unsqueeze(1)

# TRAINING DATA
X_train = torch.vstack([left_X, bottom_X, top_X, right_X])
Y_train = torch.vstack([left_Y, bottom_Y, top_Y, right_Y])
idx = np.random.choice(X_train.shape[0], Nu, replace=False)
X_train_Nu = X_train[idx, :]
Y_train_Nu = Y_train[idx, :]

lb = torch.Tensor([r[0], z[0]])
ub = torch.Tensor([r[-1], z[-1]])
X_train_Nf = lb + (ub - lb) * lhs(2, Nf)  # 2 as the inputs are x and gamma


def tot_computations(params: dict, verbose=False):
    tot_neurons = len(params['num_neurons'])
    tot_layers = len(params['num_layers'])
    tot_lr = len(params['learning_rate'])
    tot_err_func = len(params['error_func'])
    tot_activation = len(params['activation'])
    total = tot_neurons * tot_layers * tot_lr * tot_err_func * tot_activation
    if verbose:
        print(f"A total of {tot_neurons} * {tot_layers} * {tot_lr} * {tot_err_func} * {tot_activation} = {total} "
              f"parameters will be estimated. (i.e. the number of networks tried)")
    return total


def write_results(params, results, model_name, num_steps):
    layers, activation, err_func, lr = params
    epochs, train_losses, val_losses, accuracies = results
    with open(f"tuning/{model_name}.txt", 'w') as f:
        f.write(f"Layers: {layers}\n"
                f"Activation: {activation}\n"
                f"Error function: {err_func.__name__}\n"
                f"Learning rate: {lr}\n"
                f"Number of steps: {num_steps}\n"
                f"----------------------------\n"
                f"Mean relative error: {accuracies.mean()}\n"
                f"Model reference: {model_name}"
                )
        # np.save(f"tuning/{model_name}_train_loss.npy", train_losses)
        # np.save(f"tuning/{model_name}_val_loss.npy", val_losses)
        # np.save(f"tuning/{model_name}_accuracies.npy", accuracies)


def find_best_params(params: dict, num_steps=10_000):
    i = 1
    max_accuracy = {}
    for num_hidden in params['num_layers']:
        for num_neuron in params['num_neurons']:
            # Building a (1, N_neurons, ..., N_neurons, 2) net, ... num of hidden layers
            layers = np.array([2] + num_hidden * [num_neuron] + [1])
            for activation in params['activation']:
                PINN = FCN(layers, act=activation)
                # print(PINN)
                print("Architecture:", layers, "Activation:", activation)
                for err_func in params['error_func']:
                    for lr in params['learning_rate']:
                        # model_name = random_string(10)
                        model_name = generate_model_name(activation, num_hidden, num_neuron, err_func, lr)
                        if not check_model_exists(conn, model_name):
                            training = TrainingPhase(neural_net=PINN,
                                                     training_points=(X_train_Nu, Y_train_Nu, X_train_Nf),
                                                     testing_points=(X_val, Y_val), equation=pde_exponential_disc,
                                                     n_epochs=num_steps,
                                                     _loss_function=err_func)
                            print(f"Training model {i}/{tot_computations(params)}")

                            net, epochs, train_losses, val_losses, accuracies = training.train_model(
                                optimizer=torch.optim.Adam,
                                learning_rate=lr)

                            accuracy = np.abs(((Y_val - net(X_val).detach()) / Y_val).numpy())
                            insert_result(conn, (model_name, float(np.mean(accuracy)), float(np.min(accuracy)),
                                                 float(np.max(accuracy)), float(np.median(accuracy)))
                                          )
                            write_results(params=(layers, activation, err_func, lr),
                                          results=(epochs, train_losses, val_losses, accuracies),
                                          model_name=model_name, num_steps=num_steps)
                            # training.save_model(f"tuning/{model_name}_model.pt")
                            i += 1
                            max_accuracy[model_name] = np.mean(accuracy)
                            print(f"Relative error of the model for {lr=}: {np.mean(accuracy): %}")
                        else:
                            print(f"[*] Model {model_name} already exists, not training again.")
    best_model = max(max_accuracy, key=max_accuracy.get)
    best_accuracy = max_accuracy[best_model]
    print(f"Max accuracy has been achieved for model {best_model} with a minimum error of {best_accuracy}.")


def train_model_with_params(params, num_steps=10_000):
    """ Single function to train a network. Used to parallelize the code."""
    num_hidden, num_neuron, activation, err_func, lr = params

    layers = np.array([2] + num_hidden * [num_neuron] + [1])
    PINN = FCN(layers, act=activation)
    print("\n\nArchitecture:", layers, "Activation:", activation)
    model_name = generate_model_name(activation, num_hidden, num_neuron, err_func, lr)

    training = TrainingPhase(neural_net=PINN, training_points=(X_train_Nu, Y_train_Nu, X_train_Nf),
                             testing_points=(X_val, Y_val), equation=pde_exponential_disc,
                             n_epochs=num_steps,
                             _loss_function=err_func)
    net, epochs, train_losses, val_losses, accuracies = training.train_model(
        optimizer=torch.optim.Adam,
        learning_rate=lr)

    write_results(params=(layers, activation, err_func, lr),
                  results=(epochs, train_losses, val_losses, accuracies),
                  model_name=model_name, num_steps=num_steps)
    training.save_model(f"tuning/{model_name}_model.pt")
    print(f"Relative error of the model for {lr=}: {accuracies.min(): %}")
    return model_name, accuracies.min()


def find_best_params_parallelized(params: dict, num_steps=10_000):
    max_accuracy = {}

    # Créez une liste de tous les ensembles possibles d'hyperparamètres
    param_sets = [(num_hidden, num_neuron, activation, err_func, lr)
                  for num_hidden in params['num_layers']
                  for num_neuron in params['num_neurons']
                  for activation in params['activation']
                  for err_func in params['error_func']
                  for lr in params['learning_rate']]

    # Utilisez ProcessPoolExecutor pour exécuter les fonctions d'entraînement en parallèle
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(train_model_with_params, param_sets))

    # Traitez les résultats et trouvez le meilleur modèle
    for model_name, accuracy in results:
        max_accuracy[model_name] = accuracy

    best_model = max(max_accuracy, key=max_accuracy.get)
    best_accuracy = max_accuracy[best_model]
    print(f"Max accuracy has been achieved for model {best_model} with a minimum error of {best_accuracy}.")


def main():

    parameters = {
        'num_neurons': [32, 64, 128],
        'num_layers': list(range(1, 7)),
        'learning_rate': [1e-4, 1e-5, 1e-6],
        'error_func': [mse],  # , mae, rmse]
        'activation': [nn.Tanh(), nn.Sigmoid(), nn.SiLU(), nn.LogSigmoid()]

    }

    tot_computations(parameters, verbose=True)
    # find_best_params_parallelized(parameters, num_steps=10_000)
    find_best_params(parameters, num_steps=10_000)


if __name__ == '__main__':
    start = time.time()
    if DATABASE:
        db_name = "results.db"
        conn = create_connection(db_name)
        print("Creating table...")
        create_table(conn)
    main()
    tot_time = time.time() - start
    print(f"Programm finished in {tot_time} seconds.")
