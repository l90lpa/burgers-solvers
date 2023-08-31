from jax import config, jit
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from math import pi, ceil
from functools import partial
import time
import matplotlib.pyplot as plt
from mpi4py import MPI
import mpi4jax
from rk3 import RK3

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

@jit
def advection_op(u):
    n = jnp.size(u)
    dx2 = dx * 2
    u_new = jnp.empty_like(u)
    for i in range(1, n-1):
        u_new = u_new.at[i].set((u[i+1] - u[i-1]) / dx2)
    return u_new

@jit
def update_halo(y, token = None):
    sendbufl = y[ 1]
    recvbufr = y[ 1]
    recvbufr, token = mpi4jax.sendrecv(sendbufl, recvbufr, (rank + 1) % size, (rank - 1) % size, comm=comm, token=token)

    y = y.at[-1].set(recvbufr)

    sendbufr = y[-2]
    recvbufl = y[-2]
    recvbufl, token = mpi4jax.sendrecv(sendbufr, recvbufl, (rank - 1) % size, (rank + 1) % size, comm=comm, token=token)
    y = y.at[ 0].set(recvbufl)
    
    return y, token

def solver(u_initial, v, dx, num_points, dt, num_steps):

    @jit
    def f(u):
        u = - v * advection_op(u)
        u, _ = update_halo(u)
        return u
    
    return RK3(f, u_initial, dt, num_steps)
    

# Example usage
if __name__ == "__main__":
    # Define problem parameters
    l_num_points = 25
    num_points = size * l_num_points
    domain_length = 1.0
    dx = domain_length / num_points
    v = 1.2
    C = 0.1 # Courant number
    dt = (dx / v) * C  # Time step size
    num_steps = ceil(1.0 * (domain_length / (dt * v)))

    # Create the initial condition (e.g., a sinusoidal profile)
    x = jnp.linspace(0, domain_length - dx, num_points)
    u_initial = jnp.sin((2*pi/domain_length) * x)

    left_halo_point = jnp.array([u_initial[rank*l_num_points - 1]])
    domain_points = u_initial[rank*l_num_points : (rank+1)*l_num_points]
    right_halo_point = jnp.array([u_initial[((rank+1)*l_num_points)%num_points]])
    l_u_initial = jnp.concatenate((left_halo_point, domain_points, right_halo_point))

    solver_partial = lambda u0 : solver(u0, v, dx, num_points, dt, num_steps)
    
    # Run solver
    l_u_final = solver_partial(l_u_initial)
    
    u_final, _ = mpi4jax.gather(l_u_final[1:jnp.size(l_u_final)-1], 0, comm=comm)

    if rank == 0:
        u_final = u_final.flatten()
        
    # Plot the results
    if rank == 0:
        plt.plot(x, u_initial, label="Initial Condition")
        plt.plot(x, u_final, label=f"Solution after {num_steps} time steps")
        plt.xlabel("Position (x)")
        plt.ylabel("Solution (u)")
        plt.legend()
        plt.show()



