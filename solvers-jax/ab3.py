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
    y = y0

    for i in range(2, num_steps):
        y_new = y + dt * AB3_step(f, y, y_prev_1, y_prev_2)

        y_prev_2 = y_prev_1
        y_prev_1 = y
        y = y_new

    return y

def AB3_tlm(f, y0, dy0, y1, y2, dt, num_steps):

    y_prev_2 = y2
    y_prev_1 = y1
    
    dy = dy0
    y = y0

    def step(y):
        return y + dt * AB3_step(f, y, y_prev_1, y_prev_2)

    for i in range(2, num_steps):
        # y_new = y + dt * AB3_step(f, y, y_prev_1, y_prev_2)
        y_new, dy_new = jvp(step, y, dy)

        y_prev_2 = y_prev_1
        y_prev_1 = y
        y = y_new

    return y