import torch.nn as nn
from argparse import ArgumentParser
import os, datetime, inspect, random
import json
import pickle

import numpy as np

import torch
import torch.nn.functional as F

from PDESetting import PDESetting, save_config
from PDEsolvers import PDEsolver, adjoint_PDEsolver
from NNPDEML import NN, trainNNPDE, loss, trainBaseline

import matplotlib.pyplot as plt

def plot_loss_landscape_2d(model, param_name, idx1, idx2, setting, span=1.0, num_points=50):
    param_tensor = dict(model.named_parameters())[param_name]
    original_val1 = param_tensor.data[idx1].item()
    original_val2 = param_tensor.data[idx2].item()

    values1 = torch.linspace(original_val1 - span, original_val1 + span, num_points)
    values2 = torch.linspace(original_val2 - span, original_val2 + span, num_points)
    loss_grid = torch.zeros((num_points, num_points))

    meshgrid_flatten = setting["meshgrid_flatten"]
    u0 = setting["u0"]
    h = setting["h"]
    gamma = setting["gamma"]
    epsilon = setting["epsilon"]
    dt = setting["dt"]
    dx = setting["dx"]
    dy = setting["dy"]
    nt = setting["nt"]
    nx = setting["nx"]
    ny = setting["ny"]

    for i, val1 in enumerate(values1):
        for j, val2 in enumerate(values2):
            param_tensor.data[idx1] = val1
            param_tensor.data[idx2] = val2
            with torch.no_grad():
                g_eval = model(meshgrid_flatten).reshape(setting["tt"].shape)
                u = PDEsolver(u0, g_eval, gamma, epsilon, dt, dx, dy, nt, nx, ny)
                u = torch.clamp(u, min=-4, max=4)
                L = loss(u, h, dt, dx, dy).item()
                loss_grid[i, j] = L

    param_tensor.data[idx1] = original_val1
    param_tensor.data[idx2] = original_val2

    X, Y = torch.meshgrid(values1, values2, indexing='ij')
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X.numpy(), Y.numpy(), loss_grid.numpy(), levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.xlabel(f'{param_name}{idx1}')
    plt.ylabel(f'{param_name}{idx2}')
    plt.title('Loss Landscape (2D)')
    plt.tight_layout()
    plt.show()


from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

def plot_loss_landscape_3d(model, param_name, idx1, idx2, setting, span=1.0, num_points=50):
    param_tensor = dict(model.named_parameters())[param_name]
    original_val1 = param_tensor.data[idx1].item()
    original_val2 = param_tensor.data[idx2].item()

    values1 = torch.linspace(original_val1 - span, original_val1 + span, num_points)
    values2 = torch.linspace(original_val2 - span, original_val2 + span, num_points)
    loss_grid = torch.zeros((num_points, num_points))

    meshgrid_flatten = setting["meshgrid_flatten"]
    u0 = setting["u0"]
    h = setting["h"]
    gamma = setting["gamma"]
    epsilon = setting["epsilon"]
    dt = setting["dt"]
    dx = setting["dx"]
    dy = setting["dy"]
    nt = setting["nt"]
    nx = setting["nx"]
    ny = setting["ny"]

    for i, val1 in enumerate(values1):
        for j, val2 in enumerate(values2):
            param_tensor.data[idx1] = val1
            param_tensor.data[idx2] = val2
            with torch.no_grad():
                g_eval = model(meshgrid_flatten).reshape(setting["tt"].shape)
                u = PDEsolver(u0, g_eval, gamma, epsilon, dt, dx, dy, nt, nx, ny)
                L = loss(u, h, dt, dx, dy).item()
                loss_grid[i, j] = L

    param_tensor.data[idx1] = original_val1
    param_tensor.data[idx2] = original_val2

    X, Y = torch.meshgrid(values1, values2, indexing='ij')
    Z = loss_grid

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), cmap='viridis', edgecolor='none')
    ax.set_xlabel(f'{param_name}{idx1}')
    ax.set_ylabel(f'{param_name}{idx2}')
    ax.set_zlabel('Loss')
    ax.set_title('3D Loss Landscape')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":
#     PDE_name = 'HeatEquation'  # 'HeatEquation' or 'AllenCahnEquation'
#     g_target_name = 'tanh_1'  # "linear_t", "linear_x", "linear_y", "quadratic_t", "quadratic_x", "tanh_1", "tanh_2", "tanh_5", "difficult"

#     setting = PDESetting(PDE_name, g_target_name)
#     model = NN(3, 1, 1, 2/3)

#     plot_loss_landscape_2d(
#         model=model,
#         param_name="fc1.weight",
#         idx1=(0, 0),
#         idx2=(0, 2),
#         setting=setting,
#         span=10.0,
#         num_points=20
#     )


if __name__ == "__main__":

    PDE_name = 'HeatEquation'  # 'HeatEquation' or 'AllenCahnEquation'
    g_target_name = 'tanh_2'  # "linear_t", "linear_x", "linear_y", "quadratic_t", "quadratic_x", "tanh_1", "tanh_2", "tanh_5", "difficult"

    setting = PDESetting(PDE_name, g_target_name)
    model = NN(3, 5, 1, 2/3)

    with torch.no_grad():
        # First neuron: weights [-1, -1, -1], bias 1.5
        model.fc1.weight[0] = torch.tensor([-1.0, -2.0, -1.0])
        model.fc1.bias[0] = 1.5
        model.fc2.weight[0, 0] = 0.3

        # Second neuron: weights [1, -1.2, 0.5], bias -0.8
        model.fc1.weight[1] = torch.tensor([1.0, -2.4, 0.5])
        model.fc1.bias[1] = -0.8
        model.fc2.weight[0, 1] = 0.25


    plot_loss_landscape_3d(
        model=model,
        param_name="fc1.weight",
        idx1=(0, 0),
        idx2=(0, 1),
        setting=setting,
        span=20,
        num_points=21
    )