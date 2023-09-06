from jax import config, jvp, jit
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import numpy as np
import pyperf
import time


def m(u):
    dx = 0.01
    dx2 = dx * 2
    u_new = (jnp.roll(u,-1) - jnp.roll(u, 1)) / dx2
    return u_new

def TLM(u, du):
    primal, tangent = jvp(m, (u,), (du,))
    return tangent

N = 1000
rng = np.random.default_rng(12345)
z = jnp.array(rng.random((N,), dtype=np.float64))
dz = jnp.array(rng.random((N,), dtype=np.float64))

def bench_jvp(loops, z, dz):
    primals = z
    tangents = dz

    range_it = range(loops)
    start = time.perf_counter()

    for _ in range_it:
        primals, tangents = jvp(m, (primals,), (tangents,))

    end = time.perf_counter()

    return end - start

def bench_jvp_w_jit(loops, z, dz):
    primals = z
    tangents = dz

    jit_jvp = jit(lambda f, x, dx: jvp(f, x, dx), static_argnames=['f'])
    jvp_m = jit_jvp.lower(m, (primals,), (tangents,)).compile()

    range_it = range(loops)
    start = time.perf_counter()

    for _ in range_it:
        primals, tangents = jvp_m((primals,), (tangents,))

    end = time.perf_counter()

    return end - start

def bench_jvp_w_model_jit(loops, z, dz):
    primals = z
    tangents = dz

    # Duplicate model to prevent JIT cache corrupting benchmarks if the benchmark order changes
    def model_(u):
        dx = 0.01
        dx2 = dx * 2
        u_new = (jnp.roll(u,-1) - jnp.roll(u, 1)) / dx2
        return u_new
    
    model = jit(model_)
    model.lower(z).compile()

    range_it = range(loops)
    start = time.perf_counter()

    for _ in range_it:
        primals, tangents = jvp(model, (primals,), (tangents,))

    end = time.perf_counter()

    return end - start

def bench_jvp_w_jit_compilation(loops, z, dz):

    range_it = range(loops)
    total_compilation_time = 0.0
    for _ in range_it:

        jvp_wrapper = lambda f, x, dx: jvp(f, x, dx)

        start = time.perf_counter()

        compiled_jvp = jit(jvp_wrapper, static_argnames=['f']).lower(m, (z,), (dz,)).compile()

        end = time.perf_counter()
        total_compilation_time += (end - start)

        del compiled_jvp, jvp_wrapper


    return total_compilation_time

def bench_jvp_w_model_jit_compilation(loops, z):

    range_it = range(loops)
    total_compilation_time = 0.0
    for _ in range_it:

        def model_(u):
            dx = 0.01
            dx2 = dx * 2
            u_new = (jnp.roll(u,-1) - jnp.roll(u, 1)) / dx2
            return u_new
    

        start = time.perf_counter()

        compiled_model = jit(model_).lower(z).compile()

        end = time.perf_counter()
        total_compilation_time += (end - start)

        del compiled_model, model_


    return total_compilation_time

runner = pyperf.Runner()
runner.bench_time_func('Run JVP', bench_jvp, z, dz, inner_loops=1)
runner.bench_time_func('Run JVP (w/ JIT)', bench_jvp_w_jit, z, dz, inner_loops=1)
runner.bench_time_func('Run JVP (model marked for JIT)', bench_jvp_w_model_jit, z, dz, inner_loops=1)
runner.bench_time_func('Compilation JVP (w/ JIT)', bench_jvp_w_jit_compilation, z, dz, inner_loops=1)
runner.bench_time_func('Compilation JVP (model marked for JIT)', bench_jvp_w_model_jit_compilation, z, inner_loops=1)