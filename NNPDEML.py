import os
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PDEsolvers import PDEsolver, adjoint_PDEsolver


### NN setting

## NN architecture
class NN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, beta):
        super().__init__()
        self.beta = beta
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh() #nn.ReLU() #nn.Tanh() #nn.ELU() #nn.LeakyReLU(negative_slope=0.01) 
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self._initialize_weights()
    
    def _initialize_weights(self):
        self.init_params = {
            'fc1_weight': ('normal', {'mean': 0.0, 'std': 1.0}),
            'fc1_bias': ('normal', {'mean': 0.0, 'std': 1.0}),
            'fc2_weight': ('uniform', {'a': -1.0, 'b': 1.0}),
        }

        nn.init.normal_(self.fc1.weight, **self.init_params['fc1_weight'][1])
        nn.init.normal_(self.fc1.bias, **self.init_params['fc1_bias'][1])
        nn.init.uniform_(self.fc2.weight, **self.init_params['fc2_weight'][1])

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out) 
        out = 1/(self.hidden_dim)**self.beta * out
        return out


### NN training

## loss
def loss(u, h, dt, dx, dy):
    loss = torch.sum((u - h)**2) * dx*dy*dt/2
    return loss


## training logger
class TrainingLogger:
    def __init__(self):
        self.loss = []
        self.rel_loss = []
        self.rmse = []
        self.rel_rmse = []
        self.max_error = []
        self.rel_max_error = []
        self.grad_norm = []
        self.adjoint_norm = []
        self.scaled_lr = []
        self.best_loss = float('inf')
        self.best_model_state = None

    def log(self, epoch, loss_value, u, h, uhat, g, optimizer, dx, dy, dt):
        max_h = torch.max(torch.abs(h)).item()
        loss_val = loss_value.item()
        rel_loss = loss_val / (max_h ** 2)
        rmse = torch.sqrt(loss_value).item()
        rel_rmse = rmse / max_h
        max_error = torch.max(torch.abs(u - h)).item()
        rel_max_error = max_error / max_h
        adjoint_norm = torch.sum(uhat ** 2) * dx * dy * dt
        gradient_norm = max(
            (torch.norm(p.grad.data).item() for p in g.parameters() if p.grad is not None)
        )

        # Append to history
        self.loss.append(loss_val)
        self.rel_loss.append(rel_loss)
        self.rmse.append(rmse)
        self.rel_rmse.append(rel_rmse)
        self.max_error.append(max_error)
        self.rel_max_error.append(rel_max_error)
        self.grad_norm.append(gradient_norm)
        self.adjoint_norm.append(adjoint_norm.item())
        self.scaled_lr.append(optimizer.param_groups[0]['lr'])

        # Update best model
        if loss_val < self.best_loss:
            self.best_loss = loss_val
            self.best_model_state = g.state_dict().copy()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: rel loss: {rel_loss:.6e}, max error: {rel_max_error:.6e}, "
                  f"adjoint norm: {adjoint_norm:.3e}, grad norm: {gradient_norm:.3e}")

    def save(self, path):
        with open(path+'/data', 'wb') as f:
            pickle.dump({
                'loss': self.loss,
                'rel_loss': self.rel_loss,
                'max_error': self.max_error,
                'rel_max_error': self.rel_max_error,
                'grad_norm': self.grad_norm,
                'adjoint_norm': self.adjoint_norm,
                'scaled_lr': self.scaled_lr,
                'best_loss': self.best_loss,
                'best_model_state': self.best_model_state,
                'rmse': self.rmse,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path+'/data', 'rb') as f:
            data = pickle.load(f)
        obj = cls()
        obj.loss = data.get('loss', [])
        obj.rel_loss = data.get('rel_loss', [])
        obj.max_error = data.get('max_error', [])
        obj.rel_max_error = data.get('rel_max_error', [])
        obj.grad_norm = data.get('grad_norm', [])
        obj.adjoint_norm = data.get('adjoint_norm', [])
        obj.scaled_lr = data.get('scaled_lr', [])
        obj.best_loss = data.get('best_loss', float('inf'))
        obj.best_model_state = data.get('best_model_state', None)
        obj.rmse = data.get('rmse', [])
        return obj

    def plot(self, save_plot=True, path=''):
        import matplotlib.pyplot as plt
        import os

        fig, ax1 = plt.subplots(figsize=(10,6))

        color_rmse = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Relative RMSE', color=color_rmse)
        ax1.set_yscale('log')
        ax1.plot(self.rmse, color=color_rmse, label='Relative RMSE')
        ax1.tick_params(axis='y', labelcolor=color_rmse)

        ax2 = ax1.twinx()
        color_rel_max_error = 'tab:red'
        ax2.set_ylabel('Relative Max Error', color=color_rel_max_error)
        ax2.set_yscale('log')
        ax2.plot(self.rel_max_error, color=color_rel_max_error, label='Relative Max Error')
        ax2.tick_params(axis='y', labelcolor=color_rel_max_error)

        ax3 = ax1.twinx()
        color_lr = 'tab:green'
        ax3.spines['right'].set_position(('outward', 60))
        ax3.set_ylabel('Learning Rate', color=color_lr)
        ax3.set_yscale('log')
        ax3.plot(self.scaled_lr, color=color_lr, label='Learning Rate')
        ax3.tick_params(axis='y', labelcolor=color_lr)

        fig.tight_layout()
        plt.title("test")

        if save_plot:
            os.makedirs(path, exist_ok=True)
            plt.savefig(f"{path}/metrics.pdf")
        plt.close(fig)


## Neural network training
def trainNNPDE(g, num_epochs, setting, optimizer_config, scheduler_config, device):

    g = g.to(device)

    gamma = setting["gamma"]
    epsilon = setting["epsilon"]
    dx = setting["dx"]
    dy = setting["dy"]
    T = setting["T"]
    dt = setting["dt"]
    nt = setting["nt"]
    nx = setting["nx"]
    ny = setting["ny"]
    u0 = setting["u0"].to(device)
    tt = setting["tt"].to(device)
    meshgrid_flatten = setting["meshgrid_flatten"].to(device)
    h = setting["h"].to(device)

    beta = g.beta
    N = g.hidden_dim

    opt_type = optimizer_config["type"]
    opt_params = optimizer_config["params"]
    if "base_lr" in opt_params:
        base_lr = opt_params.pop("base_lr")
        opt_params["lr"] = base_lr / g.hidden_dim**(1 - 2 * g.beta)
    optimizer = getattr(torch.optim, opt_type)(g.parameters(), **opt_params)

    sched_type = scheduler_config["type"]
    sched_params = scheduler_config["params"]
    scheduler = getattr(torch.optim.lr_scheduler, sched_type)(optimizer, **sched_params)
    if hasattr(scheduler, 'patience'):
        initial_patience = scheduler.patience


    # instantiate the TrainingLogger
    logger = TrainingLogger()

    for epoch in range(num_epochs):

        optimizer.zero_grad()
        g_eval = g(meshgrid_flatten)
        g_eval = g_eval.reshape(tt.shape)

        u = PDEsolver(u0, g_eval, gamma, epsilon, dt, dx, dy, nt, nx, ny, device=device)
        u = torch.clamp(u, min=-4, max=4)  # clamp solution to avoid numerical instability

        loss_value = loss(u, h, dt, dx, dy)
                
        uhatT = torch.zeros((nx, ny), device=device)
        rhs = u-h
        uhat = adjoint_PDEsolver(uhatT, u, rhs, gamma, epsilon, T, dt, dx, dy, nt, nx, ny, device=device)

        g_eval.backward(uhat*dx*dy*dt)

        ## gradient clipping
        clipping = True
        if clipping:
            
            # allow for warm-up phase
            first_epoch = 0
            if epoch >= first_epoch:

                gt = max(torch.norm(p.grad.data).item() for p in g.parameters() if p.grad is not None)
                recent_grad_norms = logger.grad_norm[-4000:] if logger.grad_norm else [gt]
                max_recent_grad = max(recent_grad_norms)

                if epoch == first_epoch:
                    mu_t = gt
                    sigma_t = 1e-8  # avoid division by zero
                else:
                    alpha = 0.98  # smoothing factor
                    mu_t = alpha * mu_t + (1 - alpha) * gt
                    sigma_t = (alpha * sigma_t**2 + (1 - alpha) * (gt - mu_t)**2)**0.5

                z_score = (gt - mu_t) / (sigma_t + 1e-8)

                # gradient clipping using z-score
                if z_score > 0.4:
                    torch.nn.utils.clip_grad_norm_(g.parameters(), max_norm=mu_t)

                # gradient clipping using prior gradients
                torch.nn.utils.clip_grad_norm_(g.parameters(), max_norm=max_recent_grad)

        optimizer.step()

        # gradually decrease scheduler patience
        patience_decay = True
        if patience_decay:
            if hasattr(scheduler, 'patience'):
                min_patience = 0
                patience_step = 1
                steps = max(1,int(0.75*80000 / max(1, int((initial_patience)/patience_step))))
                if epoch % steps == 0:
                    if hasattr(scheduler, '_last_patience'):
                        scheduler._last_patience = scheduler._last_patience - patience_step
                    else:
                        scheduler._last_patience = scheduler.patience
                scheduler.patience = max(min_patience, int(scheduler._last_patience))
        
        scheduler.step(loss_value)
        
        # logging of error metrics using TrainingLogger
        logger.log(epoch, loss_value, u, h, uhat, g, optimizer, dx, dy, dt)

    return logger, optimizer, scheduler


## baseline computations
def trainBaseline(g, num_epochs, setting, optimizer_config, scheduler_config, device):

    g = g.to(device)

    gamma = setting["gamma"]
    epsilon = setting["epsilon"]
    dx = setting["dx"]
    dy = setting["dy"]
    T = setting["T"]
    dt = setting["dt"]
    nt = setting["nt"]
    nx = setting["nx"]
    ny = setting["ny"]
    u0 = setting["u0"].to(device)
    tt = setting["tt"].to(device)
    meshgrid_flatten = setting["meshgrid_flatten"].to(device)
    h = setting["h"].to(device)

    beta = g.beta
    N = g.hidden_dim

    opt_type = optimizer_config["type"]
    opt_params = optimizer_config["params"]
    if "base_lr" in opt_params:
        base_lr = opt_params.pop("base_lr")
        opt_params["lr"] = base_lr / g.hidden_dim**(1 - 2 * g.beta)
    optimizer = getattr(torch.optim, opt_type)(g.parameters(), **opt_params)

    sched_type = scheduler_config["type"]
    sched_params = scheduler_config["params"]
    scheduler = getattr(torch.optim.lr_scheduler, sched_type)(optimizer, **sched_params)


    # instantiate the TrainingLogger
    logger = TrainingLogger()

    # training loop for baseline computations
    # (minimize ||g-g_target||_{L^2}^2 instead of loss(u,h) = ||u-h||_{L^2}^2)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        g_eval = g(meshgrid_flatten.to(device))
        g_eval = g_eval.reshape(tt.shape)

        criterion_value = loss(g_eval, setting["g_target_eval"].to(device), dt, dx, dy)
        criterion_value.backward()

        optimizer.step()
        scheduler.step(criterion_value)

        u = PDEsolver(u0, g_eval, gamma, epsilon, dt, dx, dy, nt, nx, ny, device=device)

        uhatT = torch.zeros((nx, ny), device=device)
        rhs = u - h
        uhat = adjoint_PDEsolver(uhatT, u, rhs, gamma, epsilon, T, dt, dx, dy, nt, nx, ny, device=device)

        loss_value = loss(u, h, dt, dx, dy)

        # logging of error metrics using TrainingLogger
        logger.log(epoch, loss_value, u, h, uhat, g, optimizer, dx, dy, dt)

    return logger, optimizer, scheduler
