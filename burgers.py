import numpy as np
from scipy.linalg import toeplitz
from scipy import pi
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

def RK4(f, y0, dt, num_steps):
    y = y0

    for i in range(num_steps):
        k1 = dt * f(y)
        k2 = dt * f(y + 0.5 * k1)
        k3 = dt * f(y + 0.5 * k2)
        k4 = dt * f(y + k3)

        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y

def create_diffusion_opeartor(dx, n):

    def diffusion_op(u):
        dxdx = dx ** 2
        u_new = np.empty_like(u)
        u_new[0] = (u[n-1] -2 * u[0] + u[1]) / dxdx
        for i in range(1, n-1):
            u_new[i] = (u[i-1] - 2 * u[i] + u[i+1]) / dxdx
        u_new[n-1] = (u[n-2] - 2 * u[n-1] + u[0]) / dxdx
        return u_new
    
    return diffusion_op

def create_advection_opeartor(dx, n):
    
    def advection_op(u):
        dx2 = dx * 2
        u_new = np.empty_like(u)
        u_new[0] = (u[1] - u[n-2]) / dx2
        for i in range(1, n-1):
            u_new[i] = (u[i+1] - u[i-1]) / dx2
        u_new[n-1] = (u[1] - u[n-2]) / dx2
        return u_new
    
    return advection_op

def solver(nu, u_initial, dx, num_points, dt, num_steps):
    N = create_advection_opeartor(dx, num_points)
    L = create_diffusion_opeartor(dx, num_points)

    def f(u):
        return nu * L(u) - u * N(u)
    
    u_end = RK4(f, u_initial, dt, num_steps)
    
    return u_end


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
    u_initial[0] = 0.0
    u_initial[-1] = 0.0

    # Solve the Burgers' equation
    num_steps = 3140
    u_final = solver(nu, u_initial, dx, num_points, dt, num_steps)

    # Plot the results
    plt.plot(x, u_initial, label="Initial Velocity Profile")
    plt.plot(x, u_final, label=f"Velocity Profile after {num_steps} time steps")
    plt.xlabel("Position (x)")
    plt.ylabel("Velocity (u)")
    plt.legend()
    plt.show()
