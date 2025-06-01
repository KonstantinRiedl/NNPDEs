import os, datetime, inspect, random
import json
import pickle

import numpy as np

import torch
import torch.nn.functional as F

from PDESetting import PDESetting, save_config
from PDESolvers import PDEsolver, adjoint_PDEsolver
from NNPDEML import NN, trainNNPDE, loss, trainBaseline

# GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

#### Heat equation with NN source term

### Setting of the problem
PDE_name = 'HeatEquation'  # 'HeatEquation' or 'AllenCahnEquation'
g_target_name = 'difficult'  # "linear_t", "linear_x", "linear_y", "quadratic_t", "quadratic_x", "tanh_1", "tanh_2", "tanh_5", "difficult"

setting = PDESetting(PDE_name, g_target_name)
print(PDE_name + ' with source term ' + setting["g_target_name"])

### Numerical experiments

num_runs = 1
seed_this_run = 42

beta = [2/3]  # beta in (0.5, 1)
N = [1,2,5,10,20,50,100,200,500,1000,2000,5000]  # number of neurons in the hidden layer

num_epochs = 80000

### Optimizer and scheduler setup
## optimization setting
base_lr = 0.01

optimizer_name = "Adam"  # "Adam", "RMSprop", "SGD"

if optimizer_name == "Adam":
    optimizer_config = {
        "type": "Adam",
        "params": {
            "lr": base_lr,  # base learning rate before / g.hidden_dim**(1-2*beta_)
            "betas": [0.9, 0.999],
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False
        }
    }
elif optimizer_name == "RMSprop":
    optimizer_config = {
        "type": "RMSprop",
        "params": {
            "lr": base_lr,  # base learning rate before / g.hidden_dim**(1-2*beta_)
            "alpha": 0.9,
            "eps": 1e-08,
            "weight_decay": 0,
            "momentum": 0.0,
            "centered": False
        }
    }
elif optimizer_name == "SGD":
    optimizer_config = {
        "type": "SGD",
        "params": {
            "lr": base_lr,  # base learning rate before / g.hidden_dim**(1-2*beta_)
            "weight_decay": 0,
            "momentum": 0.0,
            "nesterov": False
        }
    }

scheduler_config = {
    "type": "ReduceLROnPlateau",
    "params": {
        "mode": "min",
        "factor": 0.95,
        "patience": 100,
        "threshold": 1e-8,
        "min_lr": 1e-8,
        "cooldown": 0
    }
}


### Technical setup
runnumber = (
    setting["PDE_name"] + "/" +
    setting["g_target_name"] + "/" +
    optimizer_config["type"] + "/" +
    datetime.date.today().strftime("%Y%b%d") + "_" +
    datetime.datetime.now().strftime("%H%M%S")
)
os.makedirs('results/' + runnumber, exist_ok=True)



## initialize result storage
# metric history
loss_history_TABLE = torch.zeros((2, len(beta), len(N), num_epochs, num_runs))
rel_loss_history_TABLE = torch.zeros((2, len(beta), len(N), num_epochs, num_runs))
max_error_history_TABLE = torch.zeros((2, len(beta), len(N), num_epochs, num_runs))
rel_max_error_history_TABLE = torch.zeros((2, len(beta), len(N), num_epochs, num_runs))

grad_norm_history_TABLE = torch.zeros((2, len(beta), len(N), num_epochs, num_runs))
adjoint_norm_history_TABLE = torch.zeros((2, len(beta), len(N), num_epochs, num_runs))
lr_history_TABLE = torch.zeros((2, len(beta), len(N), num_epochs, num_runs))
best_loss_history_TABLE = torch.zeros((2, len(beta), len(N), num_epochs, num_runs))

# result storing
best_results = {}


## loop
for j, beta_ in enumerate(beta):

    for k, N_ in enumerate(N):

        for i in range(num_runs):

            # Set random seeds for reproducibility
            seed = seed_this_run + i
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            print(f'Setting: beta={beta_}, N={N_}; Run {i+1}/{num_runs}')

            ### training

            ## NN parameters
            input_dim = 3  # for (t, x, y)
            hidden_dim = N_  # N
            output_dim = 1  # temperature

            ## NN
            g = NN(input_dim, hidden_dim, output_dim, beta_).to(device)

            ## NN for baseline
            g_baseline = NN(input_dim, hidden_dim, output_dim, beta_).to(device)
            g_baseline.load_state_dict(g.state_dict())  # create a copy of g with the same parameters

            ## optimizer and scheduler setting
            optimizer_config_run = {
                "type": optimizer_config["type"],
                "params": {
                    **{k: v for k, v in optimizer_config["params"].items() if k not in ("lr", "base_lr")},
                    "base_lr": base_lr
                }
            }

            ## NNPDE training
            logger, optimizer, scheduler = trainNNPDE(g,
                                num_epochs=num_epochs, setting=setting,
                                optimizer_config=optimizer_config_run,
                                scheduler_config=scheduler_config,
                                device=device)

            loss_history_TABLE[0,j,k,:,i] = torch.tensor(logger.loss)
            rel_loss_history_TABLE[0,j,k,:,i] = torch.tensor(logger.rel_loss)
            max_error_history_TABLE[0,j,k,:,i] = torch.tensor(logger.max_error)
            rel_max_error_history_TABLE[0,j,k,:,i] = torch.tensor(logger.rel_max_error)
            grad_norm_history_TABLE[0,j,k,:,i] = torch.tensor(logger.grad_norm)
            adjoint_norm_history_TABLE[0,j,k,:,i] = torch.tensor(logger.adjoint_norm)
            lr_history_TABLE[0,j,k,:,i] = torch.tensor(logger.scaled_lr)
            best_loss_history_TABLE[0,j,k,:,i] = torch.tensor(logger.best_loss)

            
            # save the best model state
            key = (beta_, N_)
            current_loss = logger.loss[-1]  # final loss value of the run

            if key not in best_results or current_loss < best_results[key]['loss']:
                # recompute solution with best model
                g.load_state_dict(logger.best_model_state)
                g.eval()

                meshgrid_flatten = setting["meshgrid_flatten"].to(device)
                u0 = setting["u0"].to(device)
                h = setting["h"].to(device)

                g_eval = g(meshgrid_flatten).reshape(setting["tt"].shape)
                u = PDEsolver(u0, g_eval, setting["gamma"], setting["epsilon"], setting["dt"], setting["dx"], setting["dy"], setting["nt"], setting["nx"], setting["ny"], device=device)
                rhs = u - h
                uhatT = torch.zeros((setting["nx"], setting["ny"]), device=device)
                uhat = adjoint_PDEsolver(uhatT, u, rhs, setting["gamma"], setting["epsilon"], setting["T"], setting["dt"], setting["dx"], setting["dy"], setting["nt"], setting["nx"], setting["ny"], device=device)

                best_results[key] = {
                    'loss': current_loss,
                    'g_state_dict': g.state_dict(),
                    'u': u,
                    'uhat': uhat
                }

                torch.save(g.state_dict(), os.path.join('results', runnumber, f'best_g_beta{beta_}_N{N_}.pt'))
                torch.save({'u': u, 'uhat': uhat}, os.path.join('results', runnumber, f'best_solution_beta{beta_}_N{N_}.pt'))

            # save the last model state together with the logger, optimizer and scheduler state
            torch.save({
                'g_state_dict': g.state_dict(),
                'logger': logger,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, os.path.join('results', runnumber, f'last_model_beta{beta_}_N{N_}_run{i}.pt'))


            ## baseline training
            logger_baseline, optimizer_baseline, scheduler_baseline = trainBaseline(g_baseline,
                                                   num_epochs=num_epochs,
                                                   setting=setting,
                                                   optimizer_config=optimizer_config_run,
                                                   scheduler_config=scheduler_config,
                                                   device=device)

            loss_history_TABLE[1,j,k,:,i] = torch.tensor(logger_baseline.loss)
            rel_loss_history_TABLE[1,j,k,:,i] = torch.tensor(logger_baseline.rel_loss)
            max_error_history_TABLE[1,j,k,:,i] = torch.tensor(logger_baseline.max_error)
            rel_max_error_history_TABLE[1,j,k,:,i] = torch.tensor(logger_baseline.rel_max_error)
            grad_norm_history_TABLE[1,j,k,:,i] = torch.tensor(logger_baseline.grad_norm)
            adjoint_norm_history_TABLE[1,j,k,:,i] = torch.tensor(logger_baseline.adjoint_norm)
            lr_history_TABLE[1,j,k,:,i] = torch.tensor(logger_baseline.scaled_lr)
            best_loss_history_TABLE[1,j,k,:,i] = torch.tensor(logger_baseline.best_loss)



### Save data

## save results
results = {
    'loss_history_TABLE': loss_history_TABLE,
    'rel_loss_history_TABLE': rel_loss_history_TABLE,
    'max_error_history_TABLE': max_error_history_TABLE,
    'rel_max_error_history_TABLE': rel_max_error_history_TABLE,
    'grad_norm_history_TABLE': grad_norm_history_TABLE,
    'adjoint_norm_history_TABLE': adjoint_norm_history_TABLE,
    'lr_history_TABLE': lr_history_TABLE,
    'best_loss_history_TABLE': best_loss_history_TABLE
}
torch.save(results, os.path.join('results', runnumber, 'results.pt'))


## save config
optimizer_config["params"]["base_lr"] = base_lr
if "lr" in optimizer_config["params"]:
    del optimizer_config["params"]["lr"]

save_config(
    path=f"results/{runnumber}/config.json",
    run_name=f"experiment_{runnumber}",
    run_params={"beta": beta, "N": N, "num_epochs": num_epochs, "lr": base_lr},
    model_params={"input_dim": input_dim, "hidden_dim": "hidden_dim", "output_dim": output_dim},
    pde_params={
        "gamma": setting["gamma"], "dx": setting["dx"], "dy": setting["dy"],
        "x_length": setting["x_length"], "y_length": setting["y_length"],
        "T": setting["T"], "dt": setting["dt"], "nt": setting["nt"], "nx": setting["nx"], "ny": setting["ny"]
    },
    initial_condition=setting["u0"],
    source_term=inspect.getsource(setting["g_target"]).strip(),
    optimization_params={
        "optimizer": optimizer_config,
        "scheduler": scheduler_config
    },
)
