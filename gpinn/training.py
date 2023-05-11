import torch
import numpy as np
from tqdm import trange


class TrainingPhase:
    def __init__(self, neural_net, *, training_points, testing_points, equation, n_epochs, _loss_function):
        self.neural_net = neural_net
        self.X_train_Nu, self.Y_train_Nu, self.X_train_Nf = training_points
        self.X_val, self.Y_val = testing_points
        self.pde = equation
        self.n_epochs = n_epochs
        self.loss_function = _loss_function
        self.iter = 0

    def loss_boundary(self, x_boundary, y_boundary):
        diff = y_boundary - self.neural_net.forward(x_boundary)
        return self.loss_function(diff)

    def loss_equation(self, x_pde):
        residual = self.pde(self.neural_net, x_pde)
        return self.loss_function(residual)

    def loss(self, x_boundary, y_boundary, x_pde):
        loss_bc = self.loss_boundary(x_boundary, y_boundary)
        loss_pde = self.loss_equation(x_pde)
        return loss_bc + loss_pde

    # Optimizer              X_train_Nu,Y_train_Nu,X_train_Nf
    # def closure(self, optimizer):
    #     optimizer.zero_grad()
    #     loss = self.loss(self.X_train_Nu, self.Y_train_Nu, self.X_train_Nf)
    #     loss.backward()
    #     self.iter += 1
    #     if self.iter % 100 == 0:
    #         loss2 = self.loss_boundary(self.X_test, self.Y_test)
    #         print("Training Error:", loss.detach().cpu().numpy(), "---Testing Error:", loss2.detach().numpy())
    #     return loss

    def train_model(self, optimizer, learning_rate):
        opt = optimizer(self.neural_net.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=1000,
                                                               threshold=1e-3, verbose=False)
        train_losses = []
        valid_losses = []
        epochs = []
        accuracy = []
        for epoch in trange(self.n_epochs):
            opt.zero_grad()
            loss_value = self.loss(self.X_train_Nu, self.Y_train_Nu, self.X_train_Nf)
            loss_value.backward()
            opt.step()
            scheduler.step(loss_value)

            # Compute stats on learning
            # diff = self.Y_val - self.neural_net(self.X_val).detach()
            # precision = np.abs(diff / self.Y_val)
            # loss_value_valid = self.loss_function(diff)

            # accuracy.append(precision.mean().detach())
            # valid_losses.append(loss_value_valid.item())
            train_losses.append(loss_value.item())
            epochs.append(epoch)

        return self.neural_net, np.array(epochs), np.array(train_losses), np.array(valid_losses), np.array(accuracy)

    def save_model(self, filename):
        torch.save(self.neural_net, filename)
