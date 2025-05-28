import numpy as np

import torch


#### PDE with NN source term

### PDE solver (finite differences)
# heat equation
# d_t u - gamma * (d_x^2 u + d_y^2 u) = g
# Allen Cahn equation (with epsilon > 0)
# d_t u - gamma * (d_x^2 u + d_y^2 u) + 1/epsilon^2 * (u^3 - u) = g
def PDEsolver(u0, g, gamma, epsilon, dt, dx, dy, nt, nx, ny, device=torch.device('cpu')):
    
    u = torch.zeros((nt, nx, ny), device=device)
    u[0, :, :] = u0.to(device)
    
    t = 0

    for n in range(nt-1):
        un = u[n, :, :].detach().clone().to(device)

        # nonlinearity in case of Allen Cahn equation
        nonlinearity = (un[1:-1, 1:-1]**3 - un[1:-1, 1:-1]) / (epsilon**2)

        u[n+1, 1:-1, 1:-1] = (
                un[1:-1, 1:-1]
                    + gamma * dt / dx**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])
                    + gamma * dt / dy**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
                    - dt * nonlinearity # Allen Cahn equation
                    + dt * g[n, 1:-1, 1:-1]
                    )
        
        t += dt

    return u


### Adjoint PDE solver (finite differences)
def adjoint_PDEsolver(uhatT, u, rhs, gamma, epsilon, T, dt, dx, dy, nt, nx, ny, device=torch.device('cpu')):
    
    uhat = torch.zeros((nt, nx, ny), device=device)
    uhat[-1, :, :] = uhatT.to(device)
    
    t = T

    for n in range(nt-1, 0, -1):
        uhatn = uhat[n, :, :].detach().clone().to(device)

        un = u[n, :, :].detach().clone().to(device)
        # Derivative of nonlinearity: d/du (u^3 - u) = 3u^2 - 1
        nonlinearity_adj = (3 * un[1:-1, 1:-1]**2 - 1) / (epsilon**2)

        uhat[n-1, 1:-1, 1:-1] = (
                uhatn[1:-1, 1:-1]
                    + gamma * dt / dx**2 * (uhatn[2:, 1:-1] - 2 * uhatn[1:-1, 1:-1] + uhatn[:-2, 1:-1])
                    + gamma * dt / dy**2 * (uhatn[1:-1, 2:] - 2 * uhatn[1:-1, 1:-1] + uhatn[1:-1, :-2])
                    - dt * nonlinearity_adj * uhatn[1:-1, 1:-1] # Allen Cahn equation
                    + dt * rhs[n, 1:-1, 1:-1]
                    )

        t -= dt

    return uhat



