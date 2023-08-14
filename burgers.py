import numpy as np
from scipy.linalg import toeplitz
from scipy import pi
import matplotlib.pyplot as plt
from rkm import RK4
from diff_operators import create_diffusion_opeartor, create_advection_opeartor


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
