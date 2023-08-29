from jax import config, jit, jvp
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from math import pi, ceil
import numpy as np
import matplotlib.pyplot as plt
from rk3 import *
from ab3 import *

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

def solver_rk3(u_initial, v, dx, num_points, dt, num_steps):
    A = create_advection_opeartor(dx, num_points)

    @jit
    def f(u):
        return - v * A(u)
    
    return RK3(f, u_initial, dt, num_steps)


def solver_ab3(u_initial, v, dx, num_points, dt, num_steps):
    A = create_advection_opeartor(dx, num_points)

    @jit
    def f(u):
        return - v * A(u)
    
    y2 = u_initial
    y1 = y2 + dt * f(y2)
    y0 = y1 + dt * f(y1)
    
    return AB3(f, y0, y1, y2, dt, num_steps)

def solver_ab3_tlm(u_initial, du_initial, v, dx, num_points, dt, num_steps):
    A = create_advection_opeartor(dx, num_points)

    @jit
    def f(u):
        return - v * A(u)
    
    @jit
    def step(y):
        return y + dt * f(y)

    dy2 = du_initial
    y2 = u_initial

    # y1 = y2 + dt * f(y2)
    y1, dy1 = jvp(step, y2, dy2)
    
    # y0 = y1 + dt * f(y1)
    y0, dy0 = jvp(step, y1, dy1)
    
    return AB3_tlm(f, y0, dy0, y1, y2, dt, num_steps)
    

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

    solver_partial = lambda u0 : solver_ab3(u0, v, dx, num_points, dt, num_steps)
    
    # Run solver
    u_final = solver_partial(u_initial)

    plt.plot(x, u_initial, label="Initial Condition")
    plt.plot(x, u_final, label=f"Solution after {num_steps} time steps")
    plt.xlabel("Position (x)")
    plt.ylabel("Solution (u)")
    plt.legend()
    plt.show()

