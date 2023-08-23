import numpy as np
from scipy import pi
import matplotlib.pyplot as plt
import gt4py.storage
import gt4py.cartesian.gtscript as gtscript

backend = "numpy"  # options: "numpy", "gt:cpu_ifirst", "gt:cpu_kfirst", "gt:gpu", "dace:cpu", "dace:gpu"
backend_opts = {"verbose": True} if backend.startswith("gt") else {}
dtype = np.float64
rebuild = False

@gtscript.function
def diffusion_op(dx, u):
    dxdx = dx ** 2
    u_new = (u[-1, 0, 0] - 2 * u[0, 0, 0] + u[1, 0, 0]) / dxdx
    return u_new

@gtscript.function
def advection_op(dx, u):
    dx2 = dx * 2
    u_new = (u[1, 0, 0] - u[-1, 0, 0]) / dx2
    return u_new

@gtscript.function
def f(nu, dx, u):
    return -u[0, 0, 0] * advection_op(dx, u) + nu * diffusion_op(dx, u)

@gtscript.stencil(backend=backend, rebuild=rebuild, **backend_opts)
def RK_stage(y: gtscript.Field[dtype], 
             k_prev: gtscript.Field[dtype], 
             k_next: gtscript.Field[dtype],
             y_new: gtscript.Field[dtype],
             *,
             nu: dtype, 
             dt: dtype, 
             dx: dtype,
             a: dtype,
             b: dtype):

    with computation(PARALLEL), interval(...):
        k_next = dt * f(nu, dx, y + a * k_prev)
        y_new = y_new + b * k_next / 6

@gtscript.stencil(backend=backend, rebuild=rebuild, **backend_opts)
def copy(src: gtscript.Field[dtype], 
         dst: gtscript.Field[dtype]):

    with computation(PARALLEL), interval(...):
        dst = src

def update_halo_with_boundary_conditions(y):
    y[0]  = y[-3]
    y[-1] = y[2]

def RK4(y0, dt, num_steps, nu, dx):

    nx = y0.shape[0]
    
    y = gt4py.storage.from_array(np.expand_dims([y0[-2], *y0, y0[1]], axis=(1,2)), dtype, backend=backend, aligned_index=(1, 0, 0))
    k = gt4py.storage.zeros((nx + 2, 1, 1), dtype, backend=backend, aligned_index=(1, 0, 0))
    y_new = gt4py.storage.from_array(np.expand_dims([y0[-1], *y0, y0[0]], axis=(1,2)), dtype, backend=backend, aligned_index=(1, 0, 0))
    
    for i in range(num_steps):
        RK_stage(y, k, k, y_new, nu=nu, dt=dt, dx=dx, a=0.0, b=1.0, origin=(1, 0, 0), domain=(nx, 1, 1))
        update_halo_with_boundary_conditions(k)
        RK_stage(y, k, k, y_new, nu=nu, dt=dt, dx=dx, a=0.5, b=2.0, origin=(1, 0, 0), domain=(nx, 1, 1))
        update_halo_with_boundary_conditions(k)
        RK_stage(y, k, k, y_new, nu=nu, dt=dt, dx=dx, a=0.5, b=2.0, origin=(1, 0, 0), domain=(nx, 1, 1))
        update_halo_with_boundary_conditions(k)
        RK_stage(y, k, k, y_new, nu=nu, dt=dt, dx=dx, a=1.0, b=1.0, origin=(1, 0, 0), domain=(nx, 1, 1))
        update_halo_with_boundary_conditions(k)
        update_halo_with_boundary_conditions(y_new)
        copy(y_new, y)
    
    return np.squeeze(np.asarray(y))[1:nx+1]


def solver(nu, u_initial, dx, num_points, dt, num_steps):
    
    u_end = RK4(u_initial, dt, num_steps, nu, dx)
    
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
    num_steps = 6140
    u_final = solver(nu, u_initial, dx, num_points, dt, num_steps)

    # # Plot the results
    plt.plot(x, u_initial, label="Initial Velocity Profile")
    plt.plot(x, u_final, label=f"Velocity Profile after {num_steps} time steps")
    plt.xlabel("Position (x)")
    plt.ylabel("Velocity (u)")
    plt.legend()
    plt.show()
