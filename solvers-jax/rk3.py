from jax import vjp, jvp, jit
import jax.numpy as jnp
from functools import partial

@partial(jit, static_argnames=['f', 'dt'])
def RK3_step(f, y, dt):
    k1 = f(y)
    k2 = f(y + dt * 0.5 * k1)
    k3 = f(y + dt * 0.75 * k2)

    return (1.0 / 9.0) * (2.0 * k1 + 3.0 * k2 + 4.0 * k3)

def RK3(f, y0, dt, num_steps):
    y = y0

    @jit
    def step_model(y):
        return y + dt * RK3_step(f, y, dt)
        
    for i in range(num_steps):
        y = step_model(y)

    return y

def RK3_tlm(f, y0, dy0, dt, num_steps):
    jit_jvp = jit(lambda f, x, dx: jvp(f, x, dx), static_argnames=['f'])

    y = y0
    dy = dy0

    @jit
    def step_model(y):
        return y + dt * RK3_step(f, y, dt)
    jvp_step_model = jit_jvp.lower(step_model, (y0,), (dy0,)).compile()

    for i in range(num_steps):
        y, dy = jvp_step_model((y,), (dy,))

    return y, dy

def RK3_adm(f, y0, DyN, dt, num_steps):
    def vjp_wrapper(f, primals, cotangents):
        primals, vjp_f = vjp(f, primals)
        cotangents = vjp_f((cotangents))[0]
        return primals, cotangents
    jit_vjp = jit(vjp_wrapper, static_argnames=['f'])

    @jit
    def step_model(y):
        return y + dt * RK3_step(f, y, dt)
    vjp_step_model = jit_vjp.lower(step_model, y0, DyN).compile()
    
    y_cache = jnp.zeros((num_steps, jnp.size(y0)), dtype=y0.dtype)
    y = y0


    for i in range(num_steps):
        y_cache = y_cache.at[i,:].set(y)
        y = step_model(y)

    Dy = DyN
    
    for i in range(num_steps-1, -1, -1):
        _, Dy = vjp_step_model(y_cache[i,:], Dy)

    return y, Dy