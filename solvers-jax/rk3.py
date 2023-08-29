from jax import vjp, jvp, jit
import jax.numpy as jnp
from functools import partial
import time

@partial(jit, static_argnames=['f', 'dt'])
def RK3_step(f, y, dt):
    k1 = f(y)
    k2 = f(y + dt * 0.5 * k1)
    k3 = f(y + dt * 0.75 * k2)

    return (1.0 / 9.0) * (2.0 * k1 + 3.0 * k2 + 4.0 * k3)

def RK3(f, y0, dt, num_steps):
    y = y0

    def step(y):
        return y + dt * RK3_step(f, y, dt)
    
    print("Started compilation ...")
    start = time.perf_counter()

    step_model = jit(step).lower(y).compile()
    
    end = time.perf_counter()
    print("Compilation time: ", end - start)

    print("Started steps ...")
    start = time.perf_counter()
    
    for i in range(num_steps):
        y = step_model(y)

    end = time.perf_counter()
    print("Steps [0, ", num_steps, ") total time: ", end - start)
    print("Steps [0, ", num_steps, ") average time: ", (end - start) / num_steps)

    return y

def RK3_tlm(f, y0, dy0, dt, num_steps):
    y = y0
    dy = dy0

    def step(y):
        return y + dt * RK3_step(f, y, dt)
    
    print("Started compilation ...")
    start = time.perf_counter()

    step_model = jit(step).lower(y).compile()
    step_tlm = jit(lambda y, dy: jvp(step, (y,), (dy,))).lower(y, dy).compile()
    
    end = time.perf_counter()
    print("Compliation time: ", end - start)

    print("Started steps ...")
    start = time.perf_counter()

    for i in range(num_steps):
        y_dummy, dy = step_tlm(y, dy)
        y = step_model(y)

    end = time.perf_counter()
    print("Steps [0, ", num_steps, ") total time: ", end - start)
    print("Steps [0, ", num_steps, ") average time: ", (end - start) / num_steps)

    return y, dy
