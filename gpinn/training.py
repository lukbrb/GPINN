import torch
import numpy as np
from tqdm import trange


# TODO: Adapter à n'importe quelle PDE
class TrainingPhase:
    def __init__(self, neural_net, training_points, testing_points, n_epochs, optimizer, _loss_function):
        self.neural_net = neural_net
        self.X_train_Nu, self.Y_train_Nu, self.X_train_Nf = training_points
        self.X_test, self.Y_test = testing_points
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.loss_function = _loss_function
        self.iter = 0

    def loss_boundary(self, x_boundary, y_boundary):
        return self.loss_function(self.neural_net.forward(x_boundary), y_boundary)

    def loss_equation(self, x_pde):
        _gamma = x_pde[:, 0].unsqueeze(1)
        power1 = 2 - _gamma
        power2 = 4 - _gamma
        cout = pde(self.neural_net, x_pde) - (2 * x_pde[:, 0].unsqueeze(1) ** power1) / \
               (x_pde[:, 0].unsqueeze(1) + 1) ** power2
        return self.loss_function(cout, torch.zeros_like(cout))

    def loss(self, x_boundary, y_boundary, x_pde):
        loss_bc = self.loss_boundary(x_boundary, y_boundary)
        loss_pde = self.loss_equation(x_pde)
        return loss_bc + loss_pde

    # Optimizer              X_train_Nu,Y_train_Nu,X_train_Nf
    def closure(self, optimizer):
        optimizer.zero_grad()
        loss = self.loss(self.X_train_Nu, self.Y_train_Nu, self.X_train_Nf)
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            loss2 = self.loss_boundary(self.X_test, self.Y_test)
            print("Training Error:", loss.detach().cpu().numpy(), "---Testing Error:", loss2.detach().numpy())
        return loss

    def train_model(self):
        opt = self.optimizer(self.neural_net.parameters(), lr=1e-3)

        losses = []
        epochs = []
        for epoch in trange(self.n_epochs):
            opt.zero_grad()
            loss_value = self.loss(self.X_train_Nu, self.Y_train_Nu, self.X_train_Nf)
            loss_value.backward()
            opt.step()
            losses.append(loss_value.item())
        return self.neural_net, np.array(epochs), np.array(losses)

    def save_model(self, filename):
        torch.save(self.neural_net, filename)


def pde(nn, x_pde):
    _x, _gamma = x_pde[:, 0].unsqueeze(1), x_pde[:, 1].unsqueeze(1)
    x_pde.requires_grad = True  # Enable differentiation
    f = nn(x_pde)
    f_x = torch.autograd.grad(f, x_pde, torch.ones(x_pde.shape[0], 1), retain_graph=True, create_graph=True)[0]
    f_x = f_x[:, 0].unsqueeze(1)
    func = f_x * _x ** 2
    f_xx = torch.autograd.grad(func, x_pde, torch.ones(x_pde.shape[0], 1), retain_graph=True, create_graph=True)[0]
    return f_xx
