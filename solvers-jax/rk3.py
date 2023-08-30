from jax import vjp, jvp, jit
import jax.numpy as jnp
from functools import partial
import time
from checkpoint import reverseLoopCheckpointed

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
    y = y0
    dy = dy0

    @jit
    def step_model(y):
        return y + dt * RK3_step(f, y, dt)


    for i in range(num_steps):
        y, dy = jvp(step_model, (y,), (dy,))

    return y, dy

def RK3_adm(f, y0, DyN, dt, num_steps):
    y_cache = jnp.zeros((num_steps, jnp.size(y0)), dtype=y0.dtype)
    y = y0

    @jit
    def step_model(y):
        return y + dt * RK3_step(f, y, dt)

    for i in range(num_steps):
        y_cache = y_cache.at[i,:].set(y)
        y = step_model(y)

    Dy = DyN
    
    for i in range(num_steps-1, -1, -1):
        _, vjp_fn = vjp(step_model, y_cache[i,:])
        Dy = vjp_fn(Dy)[0]

    return y, Dy

# def RK3_adm_2(f, y0, dyN, dt, num_steps):
#     y = y0
#     dy = dyN

#     @jit
#     def step_model(y):
#         return y + dt * RK3_step(f, y, dt)

#     @jit
#     def step_adm(y, dy):
#         y_dummy, vjp_fn = vjp(step_model, y)
#         return vjp_fn(dy)[0]

#     dy = reverseLoopCheckpointed(step_model, y, step_adm, dy, 8, num_steps)

#     return y, dy