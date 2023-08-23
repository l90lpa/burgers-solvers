
import numpy as np
from scipy import pi
import matplotlib.pyplot as plt
from rkm import RK4
from diff_operators import create_diffusion_opeartor, create_advection_opeartor

def solver(nu, u_initial, dx, num_points, dt, num_steps):
    N = create_advection_opeartor(dx, num_points)
    L = create_diffusion_opeartor(dx, num_points)
    def f(u):
        return - u * N(u) + nu * L(u)
    
    u_end = RK4(f, u_initial, dt, num_steps)
    
    return u_end


if __name__ == "__main__":
    # Define problem parameters
    num_points = 100
    domain_length = 2 * pi
    dx = domain_length / num_points
    nu = 0.1
    dt = 0.001  # Time step size
    num_steps = 6283

    # Create the initial condition profile (e.g., a sinusoidal profile)
    x = np.linspace(0, domain_length - dx, num_points)
    u_initial = np.sin(x)

    # Solve the Burgers' equation
    u_final = solver(nu, u_initial, dx, num_points, dt, num_steps)

    # Plot the results
    plt.plot(x, u_initial, label="Initial Condition")
    plt.plot(x, u_final, label=f"Solution after {num_steps} time steps")
    plt.xlabel("Position (x)")
    plt.ylabel("Solution (u)")
    plt.legend()
    plt.show()