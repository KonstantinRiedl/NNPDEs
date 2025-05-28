import os, datetime, inspect, json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

## NN architecture
class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, beta):
        super(NN, self).__init__()
        self.beta = beta
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh() #nn.ReLU() #nn.Tanh() #nn.ELU()
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
    

beta = 0.8  # beta in (0.5, 1)

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
dt = 0.5 * min(dx**2/(2*gamma), dy**2/(2*gamma))  # time step 
nt = int(T/dt)+1  # number of time steps
t = torch.linspace(0, T, nt)

## meshgrid
tt, xx, yy = torch.meshgrid(t, x, y, indexing='ij')
meshgrid_flatten = torch.stack([tt.flatten(), xx.flatten(), yy.flatten()], dim=1)

## NN parameters
input_dim = 3  # for (t, x, y)
# hidden_dim = 1 # N
output_dim = 1  # temperature

## NN sample
#g = NN(input_dim, hidden_dim, output_dim, beta)

with torch.no_grad():

    hidden_dim_s = [1, 4, 10, 40, 100, 400, 1000, 4000, 10000]
    runs = 100
    g_norms = torch.zeros((runs, len(hidden_dim_s)))

    for j in range(runs):
        for (i, hidden_dim) in enumerate(hidden_dim_s):
            g = NN(input_dim, hidden_dim, output_dim, beta)
            g_eval = g(meshgrid_flatten).reshape(tt.shape)
            g_norms[j, i] = torch.sqrt(torch.sum(g_eval**2)*dx*dy*dt)
            del g, g_eval  # Free memory by deleting unused tensors

    print(g_norms.mean(axis=0))

    plt.plot(hidden_dim_s, np.array(g_norms.detach()).mean(axis=0), label=r'$\|g_0^N\|_{L^2(D_T)}$')
    plt.plot(hidden_dim_s, 0.1*1/np.array(hidden_dim_s)**(beta-0.5), '--', label=r'$C/N^{\beta-1/2}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of neurons')
    plt.ylabel(r'$L_2$'+'-norm of the source term')
    plt.title('Source term norm vs. number of neurons for ' + r'$\beta$ = ' + str(beta))
    plt.legend()
    plt.grid()
    plt.show()