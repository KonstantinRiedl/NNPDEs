import inspect
import json

import numpy as np

import torch

from PDEsolvers import PDEsolver, adjoint_PDEsolver

### utility functions
def extract_lambda_source(fn):
    try:
        source_lines, _ = inspect.getsourcelines(fn)
        source = ''.join(source_lines).strip()
        return source
    except Exception:
        return "<source code not available>"

### save configuration
def save_config(path, *, run_name=None, run_params={}, model_params={}, pde_params={}, domain_params={}, initial_condition=None, source_term=None, optimization_params={}):
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        else:
            return obj

    config = {
        "run_name": run_name,
        "run_params": convert(run_params),
        "model_params": convert(model_params),
        "pde_params": convert(pde_params),
        "domain_params": convert(domain_params),
        "initial_condition": convert(initial_condition),
        "source_term": source_term if isinstance(source_term, str) else convert(source_term),
        "optimization_params": convert(optimization_params)
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


### setting of the problem
# PDE_name = 'HeatEquation' or 'AllenCahnEquation'
# g_target_name = 'linear_t', 'linear_x', 'linear_y', 'quadratic_t', 'quadratic_x', 'tanh_1', 'tanh_2', 'tanh_5' or 'difficult'
def PDESetting(PDE_name='HeatEquation', g_target_name='difficult'):
    """
    Setting of the PDE problem
    :param PDE_name: Name of the PDE problem
    :param g_target_name: Name of the target source term
    :return: Dictionary with the setting of the problem
    """

    if PDE_name == 'HeatEquation':
        epsilon = float("inf")
    elif PDE_name == 'AllenCahnEquation':
        epsilon = 0.32
    else:
        raise ValueError(f"Unknown PDE_name: {PDE_name}")


    ## PDE parameters
    gamma = 0.01  # thermal diffusivity

    ## spatial domain D
    dx, dy = .025, .025  # Grid spacing
    x_length, y_length = 0.5, 1
    x = torch.arange(0, x_length+dx, dx)
    y = torch.arange(0, y_length+dy, dy)
    nx, ny = np.shape(x)[0], np.shape(y)[0]  # Number of grid points

    ## time T
    T = 1 # time horizon
    dt = 0.5 * min(dx**2/(2*gamma), dy**2/(2*gamma))  # time step 
    nt = int(T/dt)+1  # number of time steps
    t = torch.linspace(0, T, nt)

    ## meshgrid
    tt, xx, yy = torch.meshgrid(t, x, y, indexing='ij')
    meshgrid_flatten = torch.stack([tt.flatten(), xx.flatten(), yy.flatten()], dim=1)

    ## initial temperature distribution
    u0 = torch.zeros((nx, ny))
    # Initial condition: cosine in both x and y, vanishing on boundaries
    for i in range(nx):
        for j in range(ny):
            u0[i, j] = 0.2*torch.sin(2*np.pi * x[i] / x_length) * torch.sin(2*np.pi * y[j] / y_length)

    ### target PDE solution h

    ## NN target source term
    if g_target_name == 'zero':
        g_target = lambda tt, xx, yy: 0*(tt/T)
    elif g_target_name == 'linear_t':
        g_target = lambda tt, xx, yy: (tt/T)
    elif g_target_name == 'linear_x':
        g_target = lambda tt, xx, yy: (xx/x_length)
    elif g_target_name == 'linear_y':
        g_target = lambda tt, xx, yy: (yy/y_length)
    elif g_target_name == 'quadratic_t':
        g_target = lambda tt, xx, yy: (tt/T)**2
    elif g_target_name == 'quadratic_x':
        g_target = lambda tt, xx, yy: (xx/x_length)**2
    elif g_target_name == 'tanh_1':
        g_target = lambda tt, xx, yy: torch.tanh(1.5 - (xx/x_length) - (yy/y_length) + (tt/T))
    elif g_target_name == 'tanh_2':
        g_target = lambda tt, xx, yy: (0.3 * torch.tanh(- (tt/T) - (xx/x_length) - (yy/y_length) + 1.5) + 0.25 * torch.tanh((tt/T) - 1.2*(xx/x_length) + 0.5*(yy/y_length) - 0.8))
    elif g_target_name == 'tanh_5':
        g_target = lambda tt, xx, yy: (0.3 * torch.tanh(- (tt/T) - (xx/x_length) - (yy/y_length) + 1.5) + 0.25 * torch.tanh((tt/T) - 1.2*(xx/x_length) + 0.5*(yy/y_length) - 0.8) + 0.2 * torch.tanh(-0.3*(tt/T) + 0.1*(xx/x_length) + 0.8*(yy/y_length) - 0.6) + 0.1 * torch.tanh(0.5*(tt/T) + 0.2*(xx/x_length) - 0.6*(yy/y_length) + 0.3) + 0.4 * torch.tanh(-(tt/T) + 0.3*(xx/x_length) + 0.3*(yy/y_length) - 0.5))
    elif g_target_name == 'difficult':
        g_target = lambda tt, xx, yy: 800 * ((1-(yy/y_length))**2 * ((0.2 + 0.6*(tt/T))-(yy/y_length))**2 * (yy/y_length)**2 * (1-(xx/x_length)) * (xx/x_length))
    elif g_target_name == 'dirac':
        g_target = lambda tt, xx, yy: (((yy/y_length)>(0.1 + 0.6*(tt/T))) * ((yy/y_length)<(0.2 + 0.6*(tt/T))) * ((xx/x_length)>(0.4) ) * ((xx/x_length)<(0.6)))
    else:
        raise ValueError(f"Unknown g_target_name: {g_target_name}")

    g_target_eval = g_target(tt, xx, yy)

    ## construction of target PDE solution for NN target source term
    h = PDEsolver(u0, g_target_eval, gamma, epsilon, dt, dx, dy, nt, nx, ny)

    PDESetting = {
        "PDE_name": PDE_name,
        "gamma": gamma,
        "epsilon": epsilon,
        "x_length": x_length, "y_length": y_length,
        "T": T, "dt": dt, "dx": dx, "dy": dy,
        "nt": nt, "nx": nx, "ny": ny,
        "u0": u0,
        "tt": tt, "xx": xx, "yy": yy, "meshgrid_flatten": meshgrid_flatten,
        "g_target_name": g_target_name, "g_target": g_target, "g_target_eval": g_target_eval,
        "h": h
    }

    return PDESetting