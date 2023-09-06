from jax import config, jit, jvp, vjp
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from math import pi, ceil
from functools import partial
import time
import numpy as np
import matplotlib.pyplot as plt
from ppm import ppmlin_step_advection

@jit
def step(u):
    return u + v * dt * ppmlin_step_advection(u,v,dt,dx)


def solver_ppm(u_initial, v, dx, num_points, dt, num_steps):
    u = jnp.copy(u_initial)

    for i in range(num_steps):
        u = step(u)

    return u


def solver_ppm_tlm(u_initial, du_initial, v, dx, num_points, dt, num_steps):
    jit_jvp = jit(lambda f, x, dx: jvp(f, x, dx), static_argnames=['f'])
    jvp_step = jit_jvp.lower(step, (u_initial,), (du_initial,)).compile()

    u = u_initial
    du = du_initial

    for i in range(num_steps):
        u, du = jvp_step((u,), (du,))

    return du


def solver_ppm_adm(u_initial, Du, v, dx, num_points, dt, num_steps):
    def vjp_wrapper(f, primals, cotangents):
        primals, vjp_f = vjp(f, primals)
        cotangents = vjp_f((cotangents))[0]
        return primals, cotangents
    jit_vjp = jit(vjp_wrapper, static_argnames=['f'])
    vjp_step = jit_vjp.lower(step, u_initial, Du).compile()

    u_cache = jnp.zeros((num_steps, jnp.size(u_initial)), dtype=u_initial.dtype)

    u = jnp.copy(u_initial)

    for i in range(num_steps):
        u_cache = u_cache.at[i,:].set(u)
        u = step(u)

    Du_ = Du

    for i in range(num_steps-1,-1,-1):
        _, Du_ = vjp_step(u_cache[i,:], Du_)

    Du_initial = Du_

    return Du_initial
    

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

    solver_partial = lambda u0 : solver_ppm(u0, v, dx, num_points, dt, num_steps)
    
    # Run solver
    u_final = solver_partial(u_initial)

    # Plot the results
    plt.plot(x, u_initial, label="Initial Condition")
    plt.plot(x, u_final, label=f"Solution after {num_steps} time steps")
    plt.xlabel("Position (x)")
    plt.ylabel("Solution (u)")
    plt.legend()
    plt.show()

    num_steps = 8

    def m(u):
        return solver_ppm(u, v, dx, num_points, dt, num_steps)

    def TLM(u, du):
        return solver_ppm_tlm(u, du, v, dx, num_points, dt, num_steps)

    def ADM(u, Dv):
        return solver_ppm_adm(u, Dv, v, dx, num_points, dt, num_steps)

    def testTLMLinearity(TLM, tol):
        rng = np.random.default_rng(12345)
        N = 100
        u0 = rng.random((N,), dtype=np.float64)
        du = rng.random((N,), dtype=np.float64)
        dv = np.array(TLM(jnp.array(u0), jnp.array(du)))
        dv2 = np.array(TLM(jnp.array(u0), jnp.array(2.0*du)))
        absolute_error = np.linalg.norm(dv2 - 2.0*dv)
        return absolute_error < tol, absolute_error

    def testTLMApprox(m, TLM, tol):
        rng = np.random.default_rng(12345)
        N = 100
        u0 = rng.random((N,), dtype=np.float64)
        v0 = m(jnp.array(u0))

        du = rng.random((N,), dtype=np.float64)
        dv = np.array(TLM(jnp.array(u0), jnp.array(du)))
        
        scale = 1.0

        absolute_errors = []
        relavite_errors = []
        for i in range(8):
            v1 = np.array(m(jnp.array(u0 + (scale * du))))
            absolute_error = np.linalg.norm((scale * dv) - (v1 - v0))
            absolute_errors.append(absolute_error)
            relative_error = absolute_error / np.linalg.norm(v1 - v0)
            relavite_errors.append(relative_error)
            scale /= 10.0

        # print(absolute_errors)
        # print(relavite_errors)
        min_relative_error = np.min(relavite_errors)

        return min_relative_error < tol, min_relative_error

    def testADMApprox(TLM, ADM, tol):
        rng = np.random.default_rng(12345)
        N = 100
        u0 = rng.random((N,), dtype=np.float64)
        du =  rng.random((N,), dtype=np.float64)

        dv = np.array(TLM(jnp.array(u0), jnp.array(du)))

        M = jnp.size(dv)
        Dv =  np.random.rand(M)
        Du = np.array(ADM(jnp.array(u0), jnp.array(Dv))).flatten()
        
        absolute_error = np.abs(np.dot(dv, Dv) - np.dot(du, Du))
        return absolute_error < tol, absolute_error

    print("Test TLM Linearity:")
    success, absolute_error = testTLMLinearity(TLM, 1.0e-13)
    print("success = ", success, ", absolute_error = ", absolute_error)

    print("Test TLM Approximation:")
    success, relative_error = testTLMApprox(m, TLM, 1.0e-13)
    print("success = ", success, ", relative error = ", relative_error)

    print("Test ADM Approximation:")
    success, absolute_error = testADMApprox(TLM, ADM, 1.0e-13)
    print("success = ", success, ", relative absolute_error = ", absolute_error)












