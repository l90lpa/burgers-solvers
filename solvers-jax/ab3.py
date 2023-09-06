from jax import vjp, jvp, jit
import jax.numpy as jnp
from functools import partial

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
    jit_jvp = jit(lambda f, x, dx: jvp(f, x, dx), static_argnames=['f'])

    @jit
    def step(y_prev_0, y_prev_1, y_prev_2):
        return y_prev_0 + dt * AB3_step(f, y_prev_0, y_prev_1, y_prev_2)
    jvp_step = jit_jvp.lower(step, (y0, y1, y2), (dy0, dy1, dy2)).compile()

    dy_prev_2 = dy2
    y_prev_2 = y2

    dy_prev_1 = dy1
    y_prev_1 = y1
    
    dy_prev_0 = dy0
    y_prev_0 = y0

    for i in range(2, num_steps):
        y_new, dy_new = jvp_step((y_prev_0, y_prev_1, y_prev_2), (dy_prev_0, dy_prev_1, dy_prev_2))

        dy_prev_2 = dy_prev_1
        y_prev_2 = y_prev_1
        dy_prev_1 = dy_prev_0
        y_prev_1 = y_prev_0
        dy_prev_0 = dy_new
        y_prev_0 = y_new

    return y_prev_0, dy_prev_0


def AB3_adm(f, y0, y1, y2, Dy, dt, num_steps):
    def vjp_wrapper(f, primals, cotangents):
        primals, vjp_f = vjp(f, *primals)
        cotangents = vjp_f(cotangents)
        return primals, cotangents
    jit_vjp = jit(vjp_wrapper, static_argnames=['f'])

    @jit
    def step(y_prev_0, y_prev_1, y_prev_2):
        return (y_prev_0 + dt * AB3_step(f, y_prev_0, y_prev_1, y_prev_2), y_prev_0, y_prev_1)
    vjp_step = jit_vjp.lower(step, (y0, y1, y2), (Dy, Dy, Dy)).compile()
    
    y_prev_cache = jnp.zeros((num_steps + 1, jnp.size(y0)), dtype=y0.dtype)

    y_prev_2 = y2
    y_prev_1 = y1
    y_prev_0 = y0

    y_prev_cache = y_prev_cache.at[0, :].set(y_prev_2)
    y_prev_cache = y_prev_cache.at[1, :].set(y_prev_1)
    y_prev_cache = y_prev_cache.at[2, :].set(y_prev_0)

    for i in range(2, num_steps):
        y_prev_0, y_prev_1, y_prev_2 = step(y_prev_0, y_prev_1, y_prev_2)

        y_prev_cache = y_prev_cache.at[i + 1, :].set(y_prev_0)

    Dy_prev_0 = Dy
    Dy_prev_1 = jnp.zeros_like(Dy_prev_0)
    Dy_prev_2 = jnp.zeros_like(Dy_prev_0)

    for i in range(num_steps - 1, 1, -1):
        _, cotangents = vjp_step((y_prev_cache[i - 1, :], y_prev_cache[i - 2, :], y_prev_cache[i - 2, :]), (Dy_prev_0, Dy_prev_1, Dy_prev_2))
        (Dy_prev_0, Dy_prev_1, Dy_prev_2) = cotangents

    Dy0 = Dy_prev_0
    Dy1 = Dy_prev_1
    Dy2 = Dy_prev_2

    return y_prev_0, (Dy0, Dy1, Dy2)

