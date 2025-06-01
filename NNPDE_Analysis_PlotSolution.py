import os
import json

import pandas as pd

import numpy as np

import torch

import matplotlib.pyplot as plt
import seaborn as sns

from NNPDEML import NN

from PDESetting import PDESetting

from auxiliary.plotting import plot_PDEsourceterm, plot_PDEsolution, plot_PDEadjoint, plot_PDEsourceterms, plot_PDEsolutions, plot_PDEall

### Setting
PDE_name = "HeatEquation"  # "HeatEquation" or "AllenCahnEquation"

g_target_name = "difficult"  # "linear_t", "linear_x", "linear_y", "quadratic_t", "quadratic_x", "tanh_1", "tanh_2", "tanh_5", "difficult"

optimizer_name = "Adam"  # "Adam", "RMSprop", "SGD"

# value for beta
beta = 2/3

# value for N
N = 5000

# run number
run = "001"


#######################################################
runnumber = (
    'results' + "/" +
    PDE_name + "/" +
    g_target_name + "/" +
    optimizer_name + "/" +
    run
)


### Load results

# load NN specifications from config.json
config_path = os.path.join(runnumber, 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
input_dim = config['model_params']['input_dim']
output_dim = config['model_params']['output_dim']

# load saved model and solution
g_dict = torch.load(os.path.join(runnumber, f'best_g_beta{beta}_N{N}.pt'),map_location=torch.device('cpu'))
u = torch.load(os.path.join(runnumber, f'best_solution_beta{beta}_N{N}.pt'),map_location=torch.device('cpu'))['u']
uhat = torch.load(os.path.join(runnumber, f'best_solution_beta{beta}_N{N}.pt'),map_location=torch.device('cpu'))['uhat']

g = NN(input_dim, N, output_dim, beta)
g.load_state_dict(g_dict)


# plot saved model and solution
setting = PDESetting(PDE_name, g_target_name)
plot_PDEall(
    u,
    setting["h"],
    g(setting["meshgrid_flatten"]).reshape(setting["tt"].shape),
    setting["g_target_eval"],
    N,
    save_plot=True,
    title=runnumber+f'/best_beta{beta}_N{N}'
)

plot_PDEsolution(
    setting["h"],
    N = None,
    save_plot=True,
    title=runnumber+f'/best_beta{beta}_N{N}'
)

plot_PDEsourceterm(
    setting["g_target_eval"],
    N = None,
    save_plot=True,
    title=runnumber+f'/best_beta{beta}_N{N}'
)

plot_PDEsolution(
    u,
    N = N,
    save_plot=True,
    title=runnumber+f'/best_beta{beta}_N{N}'
)

plot_PDEsourceterm(
    g(setting["meshgrid_flatten"]).reshape(setting["tt"].shape),
    N = N,
    save_plot=True,
    title=runnumber+f'/best_beta{beta}_N{N}'
)

plot_PDEadjoint(
    uhat,
    N = N,
    save_plot=True,
    title=runnumber+f'/best_beta{beta}_N{N}'
)