from jax import config, vjp, jvp, jit
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from math import pi, ceil
from functools import partial
import time
import numpy as np
import matplotlib.pyplot as plt
from checkpoint import reverseLoopCheckpointed
from rk3 import *

def create_advection_opeartor(dx, n):
    
    @jit
    def advection_op(u):
        dx2 = dx * 2
        u_new = jnp.empty_like(u)
        u_new = u_new.at[0].set((u[1] - u[n-1]) / dx2)
        for i in range(1, n-1):
            u_new = u_new.at[i].set((u[i+1] - u[i-1]) / dx2)
        u_new = u_new.at[n-1].set((u[0] - u[n-2]) / dx2)
        return u_new
    
    return advection_op

def solver(u_initial, v, dx, num_points, dt, num_steps):
    A = create_advection_opeartor(dx, num_points)

    @jit
    def f(u):
        return - v * A(u)
    
    return RK3(f, u_initial, dt, num_steps)


def solver_tlm(u_initial, du_initial, v, dx, num_points, dt, num_steps):
    A = create_advection_opeartor(dx, num_points)

    @jit
    def f(u):
        return - v * A(u)
    
    return RK3_tlm(f, u_initial, du_initial, dt, num_steps)
    

# Example usage
if __name__ == "__main__":
    # Define problem parameters
    num_points = 100
    domain_length = 1.0
    dx = domain_length / num_points
    v = 1.2
    C = 0.1 # Courant number
    dt = (dx / v) * C  # Time step size
    num_steps = ceil(1.0 * (domain_length / (dt * v)))

    # Create the initial condition (e.g., a sinusoidal profile)
    x = jnp.linspace(0, domain_length - dx, num_points)
    u_initial = jnp.sin((2*pi/domain_length) * x)
    u_initial_p = np.concatenate((u_initial[len(u_initial)-1:len(u_initial)],u_initial[:len(u_initial)-1]))

    solver_partial = lambda u0 : solver(u0, v, dx, num_points, dt, num_steps)
    
    # Run solver
    u_final = solver_partial(u_initial)

    # Plot the results
    plt.plot(x, u_initial, label="Initial Condition")
    plt.plot(x, u_final, label=f"Solution after {num_steps} time steps")
    plt.xlabel("Position (x)")
    plt.ylabel("Solution (u)")
    plt.legend()
    plt.show()

    # Run solver    
    u_final_p = solver_partial(u_initial_p)
    du_final_p = u_final_p - u_final

    # Plot the results
    plt.plot(x, u_initial_p, label="Initial Condition")
    plt.plot(x, u_final_p, label=f"Solution after {num_steps} time steps")
    plt.xlabel("Position (x)")
    plt.ylabel("Solution (u)")
    plt.legend()
    plt.show()
    
    solver_tlm_partial = lambda u0, du0: solver_tlm(u0, du0, v, dx, num_points, dt, num_steps)

    # Run solver_tlm
    du_initial = u_initial_p - u_initial
    u_final_tlm, du_final_tlm = solver_tlm_partial(u_initial, du_initial)

    # Plot the results
    plt.plot(x, u_initial, label="Initial Condition")
    plt.plot(x, u_final_tlm, label=f"Solution after {num_steps} time steps")
    plt.xlabel("Position (x)")
    plt.ylabel("Solution (u)")
    plt.legend()
    plt.show()



