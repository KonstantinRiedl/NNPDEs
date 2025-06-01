import os, datetime, inspect, json
import numpy as np

import torch
import torch.nn.functional as F

from PDESetting import save_config
from PDESetting import PDESetting
from PDESolvers import PDEsolver, adjoint_PDEsolver
from NNPDEML import loss, NN, trainNNPDE, baseline_computation

from auxiliary.plotting import plot_PDEsolution, plot_PDEsourceterm, plot_PDEsolutions, plot_PDEsourceterms, plot_PDEall

# GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#### Heat equation with NN source term

### Setting of the problem
PDE_name = 'HeatEquation'  # 'HeatEquation' or 'AllenCahnEquation'
g_target_name = 'difficult'  # "linear_t", "linear_x", "linear_y", "quadratic_t", "quadratic_x", "tanh_1", "tanh_2", "tanh_5", "difficult"

setting = PDESetting(PDE_name, g_target_name)



### 

beta = 2/3  # beta in (0.5, 1)
N = 500  # number of neurons in the hidden layer 


## NN parameters
input_dim = 3  # for (t, x, y)
hidden_dim = N  # N
output_dim = 1  # temperature



## NN training
num_epochs = 1000
lr = 0.01 / N**(1-2*beta)

optimizer_config = {
    "type": "Adam",
    "params": {
        "lr": lr,
        "betas": [0.9, 0.999],
        "eps": 1e-08,
        "weight_decay": 0,
        "amsgrad": False
    }
}

scheduler_config = {
    "type": "ReduceLROnPlateau",
    "params": {
        "mode": "min",
        "factor": 0.8,
        "patience": 10,
        "threshold": 1e-4,
        "min_lr": 1e-8,
        "cooldown": 10
    }
}

### Technical setup
runnumber = (
    setting["g_target_name"] + "/" +
    optimizer_config["type"] + "/" +
    datetime.date.today().strftime("%Y%b%d") + "_" +
    datetime.datetime.now().strftime("%H%M%S")
)
os.makedirs('results/' + runnumber, exist_ok=True)

# NNPDE 

## NN sample
g = NN(input_dim, hidden_dim, output_dim, beta).to(device)

logger = trainNNPDE(g, num_epochs=num_epochs, setting=setting,
                    optimizer_config=optimizer_config,
                    scheduler_config=scheduler_config,
                    device=device)

## construction of PDE solution for learned NN source term
u = PDEsolver(
    setting["u0"],
    g(setting["meshgrid_flatten"]).reshape(setting["tt"].shape),
    setting["gamma"], setting["dt"], setting["dx"], setting["dy"],
    setting["nt"], setting["nx"], setting["ny"]
)

plot_PDEall(
    u,
    setting["h"],
    g(setting["meshgrid_flatten"]).reshape(setting["tt"].shape),
    setting["g_target_eval"],
    save_plot=True,
    title='results/'+runnumber+'/All'
)

logger.plot(path='results/'+runnumber+'/All')
logger.save(path='results/'+runnumber+'/All')

###### ----------------- ######

# # baseline 

# ## NN sample
# g_baseline = NN(input_dim, hidden_dim, output_dim, beta).to(device)

# logger = baseline_computation(g_baseline, num_epochs=num_epochs, setting=setting,
#                              optimizer_config=optimizer_config,
#                              scheduler_config=scheduler_config,
#                              device=device)

# ## construction of PDE solution for learned NN source term
# u_baseline = PDEsolver(
#     setting["u0"],
#     g_baseline(setting["meshgrid_flatten"]).reshape(setting["tt"].shape),
#     setting["gamma"], setting["dt"], setting["dx"], setting["dy"],
#     setting["nt"], setting["nx"], setting["ny"]
# )

# # # plotting
# plot_PDEall(
#     u_baseline,
#     setting["h"],
#     g_baseline(setting["meshgrid_flatten"]).reshape(setting["tt"].shape),
#     setting["g_target_eval"],
#     save_plot=True,
#     title='results/'+runnumber+'/All_baseline'
# )

# logger.plot(path='results/'+runnumber+'/All_baseline')
# logger.save(path='results/'+runnumber+'/All_baseline')


# save_config(
#     path=f"results/{runnumber}/config.json",
#     run_name=f"experiment_{runnumber}",
#     run_params={"num_epochs": num_epochs, "lr": lr, "beta": beta},
#     model_params={"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim},
#     pde_params={
#         "gamma": setting["gamma"], "dx": setting["dx"], "dy": setting["dy"],
#         "x_length": setting["x_length"], "y_length": setting["y_length"],
#         "T": setting["T"], "dt": setting["dt"], "nt": setting["nt"],
#         "nx": setting["nx"], "ny": setting["ny"]
#     },
#     initial_condition=setting["u0"],
#     source_term=inspect.getsource(setting["g_target"]).strip()
# )
