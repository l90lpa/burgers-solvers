from jax import config, jit, jvp, vjp
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from math import pi, ceil
import matplotlib.pyplot as plt
from mpi4py import MPI
import mpi4jax
from rk3 import RK3, RK3_tlm, RK3_adm

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

@jit
def advection_op(u):
    n = jnp.size(u)
    dx2 = dx * 2
    u_new = jnp.empty_like(u)
    u_new = u_new.at[1:n-1].set((u[2:] - u[:-2]) / dx2)
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

def solver_rk3(u_initial, v, dx, num_points, dt, num_steps):
    domain_size = jnp.size(u_initial)
    ghost_domain_size = domain_size + 2

    def add_ghost_points(u_initial):
        u = jnp.zeros((ghost_domain_size,), dtype=u_initial.dtype)
        u = u.at[1:ghost_domain_size-1].set(u_initial)
        u, _ = update_halo(u)
        return u

    @jit
    def f(u):
        u = - v * advection_op(u)
        u, _ = update_halo(u)
        return u
    
    u = add_ghost_points(u_initial)
    return RK3(f, u, dt, num_steps)[1:ghost_domain_size-1]

def solver_rk3_tlm(u_initial, du_initial, v, dx, num_points, dt, num_steps):
    domain_size = jnp.size(u_initial)
    ghost_domain_size = domain_size + 2

    def add_ghost_points(u_initial):
        u = jnp.zeros((ghost_domain_size,), dtype=u_initial.dtype)
        u = u.at[1:ghost_domain_size-1].set(u_initial)
        u, _ = update_halo(u)
        return u

    u, du = jvp(add_ghost_points, (u_initial,),(du_initial,))

    @jit
    def f(u):
        u = - v * advection_op(u)
        u, _ = update_halo(u)
        return u
    
    u, du = RK3_tlm(f, u, du, dt, num_steps)
    return du[1:ghost_domain_size-1]


def solver_rk3_adm(u_initial, Du, v, dx, num_points, dt, num_steps):
    domain_size = jnp.size(u_initial)
    ghost_domain_size = domain_size + 2

    def add_ghost_points(u_initial):
        u = jnp.zeros((ghost_domain_size,), dtype=u_initial.dtype)
        u = u.at[1:ghost_domain_size-1].set(u_initial)
        u, _ = update_halo(u)
        return u
    
    u_initial_w_ghost = add_ghost_points(u_initial)

    Du_w_ghost = jnp.zeros((ghost_domain_size,), dtype=u_initial.dtype)
    Du_w_ghost = Du_w_ghost.at[1:ghost_domain_size-1].set(Du)

    @jit
    def f(u):
        u = - v * advection_op(u)
        u, _ = update_halo(u)
        return u

    u_w_ghost, Du_initial_w_ghost = RK3_adm(f, u_initial_w_ghost, Du_w_ghost, dt, num_steps)

    u = u_w_ghost[1:ghost_domain_size-1]
    _, vjp_f = vjp(add_ghost_points, u)
    Du_initial = vjp_f(Du_initial_w_ghost)[0]

    return Du_initial
    

# Example usage
if __name__ == "__main__":
    # Define problem parameters
    l_num_points = 120 // size
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

    l_u_initial = u_initial[rank*l_num_points : (rank+1)*l_num_points]

    solver_partial = lambda u0 : solver_rk3(u0, v, dx, num_points, dt, num_steps)
    
    # Run solver
    l_u_final = solver_partial(l_u_initial)
    
    u_final, _ = mpi4jax.gather(l_u_final, 0, comm=comm)

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

    mpi4jax.barrier(comm=comm)

    num_steps = 8

    def m(u):
        return solver_rk3(u, v, dx, num_points, dt, num_steps)

    def TLM(u, du):
        return solver_rk3_tlm(u, du, v, dx, num_points, dt, num_steps)

    def ADM(u, Dv):
        return solver_rk3_adm(u, Dv, v, dx, num_points, dt, num_steps)

    def alldot(a,b):
        l_dot = np.dot(a, b)
        dot, _ = mpi4jax.allreduce(l_dot, op=MPI.SUM, comm=comm)
        return dot

    def allnorm(x):
        norm = np.sqrt(alldot(x,x))
        return norm

    def testTLMLinearity(TLM, tol):
        N = 120 // size
        if rank == 0:
            rng = np.random.default_rng(12345)
            u0 = rng.random((size, N), dtype=np.float64)
            du = rng.random((size, N), dtype=np.float64)
        else:
            u0 = np.empty((N,))
            du = np.empty((N,))
            
        l_u0, _ = mpi4jax.scatter(u0, 0, comm=comm)
        l_du, _ = mpi4jax.scatter(du, 0, comm=comm)
        l_dv = np.array(TLM(jnp.array(l_u0), jnp.array(l_du)))
        l_dv2 = np.array(TLM(jnp.array(l_u0), jnp.array(2.0*l_du)))
        
        absolute_error = allnorm(l_dv2 - 2.0*l_dv)
        
        return absolute_error < tol, absolute_error

    def testTLMApprox(m, TLM, tol):
        N = 120 // size
        if rank == 0:
            rng = np.random.default_rng(12345)
            u0 = rng.random((size, N), dtype=np.float64)
            du = rng.random((size, N), dtype=np.float64)
        else:
            u0 = np.empty((N,))
            du = np.empty((N,))
            
        l_u0, _ = mpi4jax.scatter(u0, 0, comm=comm)
        l_du, _ = mpi4jax.scatter(du, 0, comm=comm)
        
        l_v0 = m(jnp.array(l_u0))
        l_dv = np.array(TLM(jnp.array(l_u0), jnp.array(l_du)))
        
        scale = 1.0

        absolute_errors = []
        relavite_errors = []
        other = []
        for i in range(15):
            l_v1 = np.array(m(jnp.array(l_u0 + (scale * l_du))))
            absolute_error = allnorm((scale * l_dv) - (l_v1 - l_v0))
            absolute_errors.append(absolute_error)
            other.append(allnorm(l_v1 - l_v0))
            relative_error = absolute_error / other[-1]
            
            relavite_errors.append(relative_error)
            scale /= 10.0

        # if rank == 0:
        #     print(absolute_errors)
        #     print(relavite_errors)
        # mpi4jax.barrier(comm=comm)
        min_relative_error = np.min(relavite_errors)

        return min_relative_error < tol, min_relative_error

    def testADMApprox(TLM, ADM, tol):
        N = 120 // size
        rng = np.random.default_rng(12345)
        if rank == 0:
            u0 = rng.random((size, N), dtype=np.float64)
            du = rng.random((size, N), dtype=np.float64)
        else:
            u0 = np.empty((N,))
            du = np.empty((N,))
            
        l_u0, _ = mpi4jax.scatter(u0, 0, comm=comm)
        l_du, _ = mpi4jax.scatter(du, 0, comm=comm)

        l_dv = np.array(TLM(jnp.array(l_u0), jnp.array(l_du)))

        M = jnp.size(l_dv)
        if rank == 0:
            Dv = rng.random((size, M), dtype=np.float64)
        else:
            Dv = np.empty((M,))
        
        l_Dv, _ = mpi4jax.scatter(Dv, 0, comm=comm)
        
        l_Du = np.array(ADM(jnp.array(l_u0), jnp.array(l_Dv))).flatten()
        
        
        absolute_error = np.abs(alldot(l_dv, l_Dv) - alldot(l_du, l_Du))
        return absolute_error < tol, absolute_error

    if rank == 0:
        print("Test TLM Linearity:")
    success, absolute_error = testTLMLinearity(TLM, 1.0e-13)
    if rank == 0:
        print("success = ", success, ", absolute_error = ", absolute_error)

    if rank == 0:
        print("Test TLM Approximation:")
    success, relative_error = testTLMApprox(m, TLM, 1.0e-13)
    if rank == 0:
        print("success = ", success, ", relative error = ", relative_error)

    if rank == 0:
        print("Test ADM Approximation:")
    success, absolute_error = testADMApprox(TLM, ADM, 1.0e-13)
    if rank == 0:
        print("success = ", success, ", absolute_error = ", absolute_error)


