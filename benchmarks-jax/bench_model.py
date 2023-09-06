from jax import config, jit
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import numpy as np
import pyperf
import time

runner = pyperf.Runner()

def model1(u):
    dx = 0.01
    dx2 = dx * 2
    n = jnp.size(u)
    u_new = jnp.empty((n,))
    for i in range(n):
        u_new = u_new.at[i].set((u[(i+1)%n] - u[(i-1)%n]) / dx2)
    return u_new

def model2(u):
    dx = 0.01
    dx2 = dx * 2
    u_new = (jnp.roll(u,-1) - jnp.roll(u, 1)) / dx2
    return u_new

N = 100
rng = np.random.default_rng(12345)
z = jnp.array(rng.random((N,), dtype=np.float64))

def bench_model1(z):
    for i in range(1):
        z = model1(z)

runner.bench_func('Run model (model w/ python loop)', bench_model1, z, inner_loops=1)

model1_compiled = jit(model1).lower(z).compile()
def bench_model1_compiled(z):
    for i in range(100):
        z = model1_compiled(z)

runner.bench_func('Run model (compiled model w/ python loop)', bench_model1_compiled, z, inner_loops=100)

def bench_model2(z):
    for i in range(100):
        z = model2(z)

runner.bench_func('Run model (model w/ no loop)', bench_model2, z, inner_loops=100)

model2_compiled = jit(model2).lower(z).compile()
def bench_model2_compiled(z):
    for i in range(100):
        z = model2_compiled(z)

runner.bench_func('Run model (compiled model w/ no loop)', bench_model2_compiled, z, inner_loops=100)

def bench_compilation_model_w_raw_loop(loops):
    N = 100
    z = jnp.empty((N,), dtype=np.float64)
    
    range_it = range(loops)
    total_compilation_time = 0.0
    
    for _ in range_it:
        # setup
        def model(u):
            dx = 0.01
            dx2 = dx * 2
            n = jnp.size(u)
            u_new = jnp.empty((n,))
            for i in range(n):
                u_new = u_new.at[i].set((u[(i+1)%n] - u[(i-1)%n]) / dx2)
            return u_new
        
        # test
        start = time.perf_counter()
        
        model_compiled = jit(model).lower(z).compile()

        end = time.perf_counter()
        total_compilation_time += (end - start)

        # teardown
        del model, model_compiled

    return total_compilation_time

runner.bench_time_func('Compile model (model w/ python loop)', bench_compilation_model_w_raw_loop, inner_loops=1)

def bench_compilation_model_no_loop(loops):
    N = 100
    z = jnp.empty((N,), dtype=np.float64)
    
    range_it = range(loops)
    total_compilation_time = 0.0
    
    for _ in range_it:
        # setup
        def model(u):
            dx = 0.01
            dx2 = dx * 2
            u_new = (jnp.roll(u,-1) - jnp.roll(u, 1)) / dx2
            return u_new
        
        # test
        start = time.perf_counter()
        
        model_compiled = jit(model).lower(z).compile()

        end = time.perf_counter()
        total_compilation_time += (end - start)

        # teardown
        del model, model_compiled

    return total_compilation_time

runner.bench_time_func('Compile model (model w/ no loop)', bench_compilation_model_no_loop, inner_loops=1)


