import os, datetime, inspect, json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from Auxiliary import plot_PDEsolution, plot_PDEsourceterm, plot_PDEsolutions, plot_PDEsourceterms, plot_PDEall

torch.set_default_dtype(torch.float64)  # float32 float64

#### Heat equation with NN source term

### Setting of the problem
gamma = 0.01  # thermal diffusivity

## spatial domain D
dx, dy = .025, .025  # Grid spacing
x_length, y_length = 0.5, 1
x = torch.arange(0, x_length+dx, dx)
y = torch.arange(0, y_length+dy, dy)
nx, ny = np.shape(x)[0], np.shape(y)[0]  # Number of grid points

## time T
T = 1 # time horizon
dt = min(dx**2/(2*gamma), dy**2/(2*gamma))/2  # time step 
nt = int(T/dt)+1  # number of time steps
t = torch.linspace(0, T, nt)

## meshgrid
tt, xx, yy = torch.meshgrid(t, x, y, indexing='ij')
meshgrid_flatten = torch.stack([tt.flatten(), xx.flatten(), yy.flatten()], dim=1)

## initial temperature distribution
u0 = torch.zeros((nx, ny))
#u0[:,0] = 1 * (1 - ((torch.linspace(-1, 1, nx))**2))
#u0[int(nx/10):int(9*nx/10), int(ny/5):int(2*ny/5)] = 1.0
#u0[:,0] = 1 * (np.sin(2*np.pi*np.linspace(-1, 1, nx)))


### PDE solver (finite differences)
# d_t u - gamma * (d_x^2 u + d_y^2 u) = g
def PDEsolver(u0, g, gamma, dt, dx, dy, nt, nx, ny):
    u = torch.zeros((nt, nx, ny))
    u[0, :, :] = u0
    t = 0

    for n in range(nt-1):
        un = u[n, :, :].detach().clone()
        u[n+1, 1:-1, 1:-1] = (
                un[1:-1, 1:-1]
                    + gamma * dt / dx**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])
                    + gamma * dt / dy**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
                    + dt * g[n, 1:-1, 1:-1]
                    )
        
        t += dt

    return u


### Adjoint PDE solver (finite differences)
def adjoint_PDEsolver(uhatT, rhs, gamma, dt, dx, dy, nt, nx, ny):
    uhat = torch.zeros((nt, nx, ny))
    uhat[-1, :, :] = uhatT
    t = T

    for n in range(nt-1, 0, -1):
        uhatn = uhat[n, :, :].detach().clone()
        uhat[n-1, 1:-1, 1:-1] = (
                uhatn[1:-1, 1:-1]
                    + gamma * dt / dx**2 * (uhatn[2:, 1:-1] - 2 * uhatn[1:-1, 1:-1] + uhatn[:-2, 1:-1])
                    + gamma * dt / dy**2 * (uhatn[1:-1, 2:] - 2 * uhatn[1:-1, 1:-1] + uhatn[1:-1, :-2])
                    + dt * rhs[n, 1:-1, 1:-1]
                    )

        t -= dt

    return uhat


### target PDE solution h

## NN target source term
#g_target = lambda tt, xx, yy: 50*(torch.abs(0.5*x_length-xx)<0.4)*(torch.abs((0.5+0.4*tt/T)*y_length-yy)<0.4)
#g_target = lambda tt, xx, yy: (torch.abs((0.5+0.4*tt/T)*y_length-yy)<0.4)
#g_target = lambda tt, xx, yy: 50*(xx/x_length) + 50*(yy/y_length)
g_target = lambda tt, xx, yy: (xx/x_length)
#g_target = lambda tt, xx, yy: 50*(xx/x_length)**2
#g_target = lambda tt, xx, yy: (tt/T)
g_target_eval = g_target(tt, xx, yy)


## construction of target PDE solution for NN target source term
h = PDEsolver(u0, g_target_eval, gamma, dt, dx, dy, nt, nx, ny)

### NN setting

beta = 0.9  # beta in (0.5, 1)

## NN architecture
class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, beta):
        super(NN, self).__init__()
        self.beta = beta
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU() #nn.ReLU() #nn.Tanh() #nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.fc1.bias, mean=0.0, std=1.0)
        nn.init.uniform_(self.fc2.weight, a=-1.0, b=1.0)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out) 
        out = 1/(torch.tensor(self.hidden_dim))**self.beta * out
        return out



### NN training

## loss
def loss(u, h, dt=dt, dx=dx, dy=dy):
    loss = torch.sum((u - h)**2)*dx*dy*dt/2
    return loss

eps_range = 10**-np.linspace(0, 8, 50)

def trainNNPDE(g):
    ## Training the neural network
    num_epochs = 1

    # optimizer = torch.optim.Adam(g.parameters(), lr=0.1)
    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()
    #     g_eval = g(meshgrid_flatten)
    #     g_eval = g_eval.reshape(tt.shape)
    #     u = PDEsolver(u0, g_eval, gamma, dt, dx, dy, nt, nx, ny, plot=False)
    #     loss_value = loss(u, h)
    #     loss_value.backward()
    #     optimizer.step()

    #     print(f'Epoch {epoch}, Loss: {loss_value.item()}')

    N = g.hidden_dim
    lr = 0.1 / N**(1-2*beta)

    grad_error = torch.zeros(num_epochs,len(eps_range),len(list(g.parameters())))
    
    #optimizer = torch.optim.SGD(g.parameters(), lr=lr)
    optimizer = torch.optim.RMSprop(g.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        g_eval = g(meshgrid_flatten)
        g_eval = g_eval.reshape(tt.shape)

        u = PDEsolver(u0, g_eval, gamma, dt, dx, dy, nt, nx, ny)
        
        loss_value = loss(u, h, dt, dx, dy)
        
        uhatT = torch.zeros((nx, ny))
        rhs = u-h
        uhat = adjoint_PDEsolver(uhatT, rhs, gamma, dt, dx, dy, nt, nx, ny)

        g_eval.backward(uhat*dx*dy*dt)
        
        with torch.no_grad():

                j = 0
                for eps in eps_range:
                    k = 0
                    
                    print(epoch, j, k)

                    for param in g.parameters():

                        param_flat = param.view(-1)
                        finite_diff_grad = torch.zeros_like(param_flat, dtype=torch.float64)

                        for i in range(param_flat.size(0)):
                            g_perturbed = NN(input_dim, hidden_dim, output_dim, beta)
                            g_perturbed.load_state_dict(g.state_dict())

                            # Perturb parameter
                            param_flat[i] += eps  # Apply perturbation
                            g_perturbed.load_state_dict(g.state_dict())  # Update g_perturbed with the perturbed parameter
                            g_perturbed_eval = g_perturbed(meshgrid_flatten)
                            g_perturbed_eval = g_perturbed_eval.reshape(tt.shape)
                            u_perturbed = PDEsolver(u0, g_perturbed_eval, gamma, dt, dx, dy, nt, nx, ny)
                            loss_perturbed_value = loss(u_perturbed, h, dt, dx, dy)

                            # Revert perturbation
                            param_flat[i] -= eps
                            g_perturbed.load_state_dict(g.state_dict())  # Update g_perturbed with the perturbed parameter


                            # Perturb parameter
                            param_flat[i] -= eps  # Apply perturbation
                            g_perturbed.load_state_dict(g.state_dict())  # Update g_perturbed with the perturbed parameter
                            g_perturbed_eval2 = g_perturbed(meshgrid_flatten)
                            g_perturbed_eval2 = g_perturbed_eval2.reshape(tt.shape)
                            u_perturbed2 = PDEsolver(u0, g_perturbed_eval2, gamma, dt, dx, dy, nt, nx, ny)
                            loss_perturbed_value2 = loss(u_perturbed2, h, dt, dx, dy)

                            # Revert perturbation
                            param_flat[i] += eps
                            g_perturbed.load_state_dict(g.state_dict())  # Update g_perturbed with the perturbed parameter


                            # Compute finite difference gradient
                            finite_diff_grad[i] = (loss_perturbed_value - loss_perturbed_value2) / (2*eps)  # central finite difference
                            finite_diff_grad[i] = (loss_perturbed_value - loss_value) / eps  # forward finite difference


                        # Reshape finite difference gradient back to parameter shape
                        finite_diff_grad = finite_diff_grad.view(param.shape)

                        # Print finite difference gradient for comparison
                        print(f"Difference:\n{torch.norm(finite_diff_grad - param.grad)}")

                        print(torch.norm(finite_diff_grad))

                        # Store the finite difference gradient in the grad_error tensor
                        grad_error[epoch, j, k] = (torch.norm(finite_diff_grad - param.grad)).detach()  # absolute error
                        grad_error[epoch, j, k] = (torch.norm(finite_diff_grad - param.grad) / torch.norm(param.grad)).detach()  # relative error
                        k += 1

                    j += 1
            
            #param.grad = finite_diff_grad

        
        optimizer.step()

        if epoch == 0:
            loss_history = []
        loss_history.append(loss_value.item())

    return loss_history, grad_error



## NN parameters
input_dim = 3  # (t, x, y)
hidden_dim = 20 # N
output_dim = 1  # Temperature

## NN sample
g = NN(input_dim, hidden_dim, output_dim, beta)



## NN training
loss_history, grad_error = trainNNPDE(g)


## construction of target PDE solution for NN target source term
u = PDEsolver(u0, g(meshgrid_flatten).reshape(tt.shape), gamma, dt, dx, dy, nt, nx, ny)


print(grad_error[0,:,0])

plt.plot(eps_range, np.array(grad_error[0,:,0]))
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.show()