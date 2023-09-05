from jax import config, jit
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from math import pi
from functools import partial
import time
import numpy as np
import matplotlib.pyplot as plt

def create_ab3(y1, y2):
    def AB3(f, y0, dt, num_steps):
        y_prev = [y1, y2]
        y = y0

        for i in range(num_steps):
            y_new = y + (dt / 12) * (23 * f(y) - 16 * f(y_prev[0]) + 5 * f(y_prev[1]))

            y_prev[1] = y_prev[0]
            y_prev[0] = y
            y = y_new

        return y

    return AB3

@partial(jit, static_argnames=['f'])
def RK4_step(f, y, dt):
    k1 = f(y)
    k2 = f(y + dt * 0.5 * k1)
    k3 = f(y + dt * 0.5 * k2)
    k4 = f(y + dt * k3)

    return (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def RK4(f, y0, dt, num_steps):
    y = y0

    start = time.perf_counter()

    y = y + dt * RK4_step(f, y, dt)
    
    end = time.perf_counter()
    print("First step: ", end - start)

    start = time.perf_counter()
    
    for i in range(1, num_steps):
        y = y + dt * RK4_step(f, y, dt)
    
    end = time.perf_counter()
    
    if num_steps > 1:
        print("Steps [1, ", num_steps, "): ", end - start)
        print("Step average [1, ", num_steps, "): ", (end - start) / (num_steps - 1))

    return y

def create_diffusion_opeartor(dx, n):

    @jit
    def diffusion_op(u):
        dxdx = dx ** 2
        u_new = (jnp.roll(u, -1) - 2.0 * u + jnp.roll(u, 1)) / dxdx
        return u_new
    
    return diffusion_op

def create_advection_opeartor(dx, n):
    
    @jit
    def advection_op(u):
        dx2 = dx * 2
        u_new = (jnp.roll(u, -1) - jnp.roll(u, 1)) / dx2
        return u_new
    
    return advection_op

def solver(u_initial, nu, dx, num_points, dt, num_steps):
    A = create_advection_opeartor(dx, num_points)
    D = create_diffusion_opeartor(dx, num_points)

    @jit
    def f(u):
        return - u * A(u) + nu * D(u)

    return RK4(f, u_initial, dt, num_steps)
    

# Example usage
if __name__ == "__main__":
    # Define problem parameters
    num_points = 100
    domain_length = 2 * pi
    dx = domain_length / (num_points - 1)
    nu = 0.1
    dt = 0.001  # Time step size
    num_steps = 3141

    # Create the initial velocity profile (e.g., a sinusoidal profile)
    x = jnp.linspace(0, domain_length - dx, num_points)
    u_initial = jnp.sin(x)

    solver_partial = lambda u0 : solver(u0, nu, dx, num_points, dt, num_steps)
    
    # Solve the Burgers' equation
    u_final = solver_partial(u_initial)

    # Plot the results
    plt.plot(x, u_initial, label="Initial Velocity Profile")
    plt.plot(x, u_final, label=f"Velocity Profile after {num_steps} time steps")
    plt.xlabel("Position (x)")
    plt.ylabel("Velocity (u)")
    plt.legend()
    plt.show()



