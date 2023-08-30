from jax import vjp, jvp, jit
import jax.numpy as jnp
from functools import partial
import time
from checkpoint import reverseLoopCheckpointed

@partial(jit, static_argnames=['f'])
def AB3_step(f, y, y1, y2):
    return (1.0 / 12.0) * (23.0 * f(y) - 16.0 * f(y1) + 5.0 * f(y2))


def AB3(f, y0, y1, y2, dt, num_steps):

    y_prev_2 = y2
    y_prev_1 = y1
    y_prev_0 = y0

    for i in range(2, num_steps):
        y_new = y_prev_0 + dt * AB3_step(f, y_prev_0, y_prev_1, y_prev_2)

        y_prev_2 = y_prev_1
        y_prev_1 = y_prev_0
        y_prev_0 = y_new

    return y_prev_0


def AB3_tlm(f, y0, y1, y2, dy0, dy1, dy2, dt, num_steps):

    @jit
    def step(y_prev_0, y_prev_1, y_prev_2):
        return y_prev_0 + dt * AB3_step(f, y_prev_0, y_prev_1, y_prev_2)

    dy_prev_2 = dy2
    y_prev_2 = y2

    dy_prev_1 = dy1
    y_prev_1 = y1
    
    dy_prev_0 = dy0
    y_prev_0 = y0

    for i in range(2, num_steps):
        # y_new = y + dt * AB3_step(f, y, y_prev_1, y_prev_2)
        y_new, dy_new = jvp(step, (y_prev_0, y_prev_1, y_prev_2), (dy_prev_0, dy_prev_1, dy_prev_2))

        dy_prev_2 = dy_prev_1
        y_prev_2 = y_prev_1
        dy_prev_1 = dy_prev_0
        y_prev_1 = y_prev_0
        dy_prev_0 = dy_new
        y_prev_0 = y_new

    return y_prev_0, dy_prev_0


def AB3_adm(f, y0, y1, y2, Dy, dt, num_steps):

    @jit
    def step(y_prev_0, y_prev_1, y_prev_2):
        return y_prev_0 + dt * AB3_step(f, y_prev_0, y_prev_1, y_prev_2)
    
    y_prev_2_cache = jnp.zeros((num_steps, jnp.size(y0)), dtype=y0.dtype)
    y_prev_1_cache = jnp.zeros_like(y_prev_2_cache)
    y_prev_0_cache = jnp.zeros_like(y_prev_2_cache)
    y_prev_2_cache = y_prev_2_cache.at[0, :].set(y2)
    y_prev_1_cache = y_prev_1_cache.at[0, :].set(y1)
    y_prev_0_cache = y_prev_0_cache.at[0, :].set(y0)

    y_prev_2 = y2
    y_prev_2_cache = y_prev_2_cache.at[1, :].set(y_prev_2)
    y_prev_1 = y1
    y_prev_1_cache = y_prev_1_cache.at[1, :].set(y_prev_1)
    y_prev_0 = y0
    y_prev_0_cache = y_prev_0_cache.at[1, :].set(y_prev_0)

    for i in range(2, num_steps):
        y_new = step(y_prev_0, y_prev_1, y_prev_2)

        y_prev_2 = y_prev_1
        y_prev_2_cache = y_prev_2_cache.at[i, :].set(y_prev_2)
        y_prev_1 = y_prev_0
        y_prev_1_cache = y_prev_1_cache.at[i, :].set(y_prev_1)
        y_prev_0 = y_new
        y_prev_0_cache = y_prev_0_cache.at[i, :].set(y_prev_0)

    Dy_prev_0 = Dy
    Dy_new = 0
    Dy_prev_1 = 0
    Dy_prev_2 = 0

    for i in range(num_steps - 1, 1, -1):
        Dy_new += Dy_prev_0
        Dy_prev_0 += Dy_prev_1
        Dy_prev_1 += Dy_prev_2

        _, vjp_step = vjp(step, y_prev_0_cache[i - 1, :], y_prev_1_cache[i - 1, :], y_prev_2_cache[i - 1, :])
        tmp0, tmp1, tmp2 = vjp_step(Dy_new)
        Dy_prev_0 += tmp0
        Dy_prev_1 += tmp1
        Dy_prev_2 += tmp2

    Dy0 = Dy_prev_0
    Dy1 = Dy_prev_1
    Dy2 = Dy_prev_2

    return y_prev_0, (Dy0, Dy1, Dy2)

def AB3_adm_2(f, y0, y1, y2, Dy, dt, num_steps):

    @jit
    def step(y_prev_0, y_prev_1, y_prev_2):
        return y_prev_0 + dt * AB3_step(f, y_prev_0, y_prev_1, y_prev_2)

    y_prev_2 = y2
    y_prev_1 = y1
    y_prev_0 = y0

    for i in range(2, num_steps):
        y_new = step(y_prev_0, y_prev_1, y_prev_2)
        y_prev_2 = y_prev_1
        y_prev_1 = y_prev_0
        y_prev_0 = y_new

    Dy_prev_0 = Dy
    Dy_prev_1 = 0 
    Dy_prev_2 = 0
    Dy_new = 0

    for i in range(num_steps-1, 1, -1):
        # y_prev_0 = y_new
        Dy_new += Dy_prev_0
        # y_prev_1 = y_prev_0
        Dy_prev_0 += Dy_prev_1
        # y_prev_2 = y_prev_1
        Dy_prev_1 += Dy_prev_2
        # y_new = step(y_prev_0, y_prev_1, y_prev_2)
        _, vjp_step = vjp(step, y_prev_0_cache[i - 1, :], y_prev_1_cache[i - 1, :], y_prev_2_cache[i - 1, :])
        tmp0, tmp1, tmp2 = vjp_step(Dy_new)
        Dy_prev_0 += tmp0
        Dy_prev_1 += tmp1
        Dy_prev_2 += tmp2

    y_prev_2 = y2
    y_prev_1 = y1
    y_prev_0 = y0

    return y_prev_0
