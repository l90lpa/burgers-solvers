import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from ppm_advection import ppmlin_step_advection
from rkm import RK3_step
from diff_operators import *
from math import ceil, floor, pi

def solver(v, nu, eps, u_initial, dx, num_points, dt, num_steps, solver_option):
    u = np.copy(u_initial)

    if solver_option == 1:    
        D = create_diffusion_opeartor(dx, num_points)
        
        def f(u):
            return nu * D(u)
        
        for i in range(num_steps):
            u = u + dt * ppmlin_step_advection(u,v,dt,dx,num_points) + dt * RK3_step(f, u, dt)

    elif solver_option == 2:
        A = create_advection_opeartor(dx, num_points)
        D = create_diffusion_opeartor(dx, num_points)
        D2 = create_fourth_order_diffusion_operator(dx, num_points, eps)
        
        def f(u):
            return - v * A(u) + nu * D(u) + D2(u)
        
        for i in range(num_steps):
            u = u + dt * RK3_step(f, u, dt)

    elif solver_option == 3:
        A = create_advection_opeartor(dx, num_points)
        D = create_diffusion_opeartor(dx, num_points)
        D2 = create_fourth_order_diffusion_matrix(dx, num_points, eps)
        I = sparse.identity(num_points, dtype=np.float64)
        
        def f(u):
            return - v * A(u) + nu * D(u)
        
        L = I - (dt * 0.5 * D2)
        B = I + (dt * 0.5 * D2)
        
        for i in range(num_steps):
            b = (B * u) + dt * RK3_step(f, u, dt)
            u = sparse.linalg.spsolve(L, b)
            
    
    return u


def grid_setup(dx,N):

    x = np.empty((N,), dtype=np.float64)

    for i in range(N):
        x[i] = (i + 1) * dx

    return x

def initial_condition(x,N,option):
        
    # Options:
    # option==1 sine function
    # option==2 step function
    # option==3 one cloud
    # option==4 many clouds

    u0 = np.zeros((N,), dtype=np.float64)
        
    if (option==1):
        for i in range(N):
            u0[i]=0.5*(1.0+np.sin(2.0*pi*x[i]))

    elif (option==2):
        for i in range(N):
            if (x[i] > 0.25 and x[i] < 0.75):
                u0[i]=1.0

    elif (option==3):
        u0[floor(N/2.0)]=1.0

    elif (option==4):
        u0[floor(4.0*N/32.0)]=1.0
        u0[floor(5.0*N/32.0)]=1.0
        u0[floor(10.0*N/32.0)]=1.0
        u0[floor(14.0*N/32.0)]=1.0
        u0[floor(16.0*N/32.0)]=1.0
        u0[floor(18.0*N/32.0)]=1.0
        u0[floor(26.0*N/32.0)]=1.0
        u0[floor(29.0*N/32.0)]=1.0

    return u0


if __name__ == "__main__":
    
    # Define problem parameters
    domain_length = 1.0
    num_points = 128
    dx = domain_length / num_points     # Space step size
    c = 0.1                             # Courant number
    dt = c * dx                       # Time step size
    # dt = (2.5/4) * dx ** 2              # Time step size
    num_steps = ceil(1.0 * (1.0/(dt)))
    v = 1.0                             # Advection coefficient
    nu = 0.0005                         # Diffusion coefficient
    eps = 0.0000005                     # 4th-order diffusion coefficient
    solver_option = 1                   # Choose between solver schemes
    ic_option = 2                       # Chosen initial condition

    print("domain_length = ", domain_length)
    print("num_points = ", num_points)
    print("dx = ", dx)
    print("c = ", c)
    print("dt = ", dt)
    print("num_steps = ", num_steps)
    print("v = ", v)
    print("nu = ", nu)
    print("eps = ", eps)
    print("solver_option = ", solver_option)
    print("ic_option = ", ic_option)

    x = grid_setup(dx,num_points)
    
    u_initial = initial_condition(x,num_points,ic_option)

    # Solve the Burgers' equation
    u_final = solver(v, nu, eps, u_initial, dx, num_points, dt, num_steps, solver_option)

    # Plot the results
    plt.plot(x, u_initial, label="Initial Velocity Profile")
    plt.plot(x, u_final, label=f"Velocity Profile after {num_steps} time steps")
    plt.xlabel("Position (x)")
    plt.ylabel("Velocity (u)")
    plt.legend()
    plt.show()