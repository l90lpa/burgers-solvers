import numpy as np
from scipy.linalg import toeplitz
from scipy import pi
import matplotlib.pyplot as plt
from ppm_advection import ppmlin_step_advection, ppmunlimited_step_advection  
from rkm import RK4_step
from diff_operators import create_diffusion_opeartor
from math import ceil, floor, pi

def solver(nu, u_initial, dx, num_points, dt, num_steps):
    u = np.copy(u_initial)

    D = create_diffusion_opeartor(dx, num_points)
    
    def nu_D(u):
        return nu * D(u)

    u_new = np.empty((num_points,), dtype=np.float64)

    for i in range(num_steps):
        # u_new = u + dt * u * ppmlin_step_advection(u,1.0,dt,dx,num_points) + dt * RK4_step(nu_D, u, dt)
        u_new = u + dt * ppmlin_step_advection(u,1.0,dt,dx,num_points) #+ dt * RK4_step(nu_D, u, dt)
        u=u_new
    
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
    num_points = 64
    dx = domain_length / num_points     # Space step size
    c = 0.1                             # Courant number
    dt = c * dx                         # Time step size
    num_steps = ceil(0.5 * (1.0/(dt)))
    nu = 0.005                          # Diffusion coefficient
    ic_option = 2                       # Chosen initial condition

    print("domain_length = ", domain_length)
    print("num_points = ", num_points)
    print("dx = ", dx)
    print("c = ", c)
    print("dt = ", dt)
    print("num_steps = ", num_steps)
    print("nu = ", nu)
    print("ic_option = ", ic_option)

    x = grid_setup(dx,num_points)
    
    u_initial = initial_condition(x,num_points,ic_option)

    # Solve the Burgers' equation
    u_final = solver(nu, u_initial, dx, num_points, dt, num_steps)

    # Plot the results
    plt.plot(x, u_initial, label="Initial Velocity Profile")
    plt.plot(x, u_final, label=f"Velocity Profile after {num_steps} time steps")
    plt.xlabel("Position (x)")
    plt.ylabel("Velocity (u)")
    plt.legend()
    plt.show()