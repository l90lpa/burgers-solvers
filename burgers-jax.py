import jax.numpy as np
from jax.scipy.linalg import toeplitz
from scipy import pi
from jax import vjp
from jax import jit
from functools import partial
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
    k1 = dt * f(y)
    k2 = dt * f(y + 0.5 * k1)
    k3 = dt * f(y + 0.5 * k2)
    k4 = dt * f(y + k3)

    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def RK4(f, y0, dt, num_steps):
    y = y0

    for i in range(num_steps):
        y = RK4_step(f, y, dt)

    return y

def create_diffusion_opeartor(dx, n):

    @jit
    def diffusion_op(u):
        dxdx = dx ** 2
        u_new = np.empty_like(u)
        u_new = u_new.at[0].set((u[n-1] -2 * u[0] + u[1]) / dxdx)
        for i in range(1, n-1):
            u_new = u_new.at[i].set((u[i-1] - 2 * u[i] + u[i+1]) / dxdx)
        u_new = u_new.at[n-1].set((u[n-2] - 2 * u[n-1] + u[0]) / dxdx)
        return u_new
    
    return diffusion_op

def create_advection_opeartor(dx, n):
    
    @jit
    def advection_op(u):
        dx2 = dx * 2
        u_new = np.empty_like(u)
        u_new = u_new.at[0].set((u[1] - u[n-1]) / dx2)
        for i in range(1, n-1):
            u_new = u_new.at[i].set((u[i+1] - u[i-1]) / dx2)
        u_new = u_new.at[n-1].set((u[0] - u[n-2]) / dx2)
        return u_new
    
    return advection_op

def solver(u_initial, nu, dx, num_points, dt, num_steps):
    A = create_advection_opeartor(dx, num_points)
    D = create_diffusion_opeartor(dx, num_points)

    @jit
    def f(u):
        return nu * D(u) - u * A(u)
    
    u_end = RK4(f, u_initial, dt, num_steps)
    
    return u_end

# Example usage
if __name__ == "__main__":
    # Define problem parameters
    num_points = 100
    domain_length = 2 * pi
    dx = domain_length / (num_points - 1)
    nu = 0.1
    dt = 0.001  # Time step size

    # Create the initial velocity profile (e.g., a sinusoidal profile)
    x = np.linspace(0, domain_length, num_points)
    u_initial = np.sin(x)

    # Solve the Burgers' equation
    num_steps = 3140
    u_final = np.empty_like(u_initial)
    u_final = solver(u_initial, nu, dx, num_points, dt, num_steps)

    # Plot the results
    plt.plot(x, u_initial, label="Initial Velocity Profile")
    plt.plot(x, u_final, label=f"Velocity Profile after {num_steps} time steps")
    plt.xlabel("Position (x)")
    plt.ylabel("Velocity (u)")
    plt.legend()
    plt.show()

    # solver_partial = partial(solver, nu=nu, dx=dx, num_points=num_points, dt=dt, num_steps=num_steps)

    # du_initial = np.zeros_like(u_initial)
    # du_initial = du_initial.at[0].set(1)
    # primals, solver_vjp = vjp(solver_partial, u_initial)
