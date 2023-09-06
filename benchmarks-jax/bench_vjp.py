from jax import config, vjp, jit
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

def ADM(u, du):
    primal, adjoint_f = vjp(m, u)
    return adjoint_f((du))


N = 1000
rng = np.random.default_rng(12345)
z = jnp.array(rng.random((N,), dtype=np.float64))
Dz = jnp.array(rng.random((N,), dtype=np.float64))


def bench_vjp(loops, z, Dz):
    primals = z
    cotangents = Dz

    range_it = range(loops)
    start = time.perf_counter()

    for _ in range_it:
        primals, vjp_m = vjp(m, primals)
        cotangents = vjp_m((cotangents))[0]

    end = time.perf_counter()

    return end - start

def bench_vjp_w_jit(loops, z, Dz):
    primals = z
    cotangents = Dz

    def vjp_wrapper(f, primals, cotangents):
        primals, vjp_f = vjp(f, primals)
        cotangents = vjp_f((cotangents))[0]
        return primals, cotangents

    jit_vjp = jit(vjp_wrapper, static_argnames=['f'])
    vjp_m = jit_vjp.lower(m, primals, cotangents).compile()

    range_it = range(loops)
    start = time.perf_counter()

    for _ in range_it:
        primals, cotangents = vjp_m(primals, cotangents)

    end = time.perf_counter()

    return end - start

def bench_vjp_w_model_jit(loops, z, Dz):
    primals = z
    cotangents = Dz

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
        primals, vjp_m = vjp(model, primals)
        cotangents = vjp_m((cotangents))[0]

    end = time.perf_counter()

    return end - start

def bench_vjp_w_jit_compilation(loops, z, Dz):

    range_it = range(loops)
    total_compilation_time = 0.0
    for _ in range_it:

        def vjp_wrapper(f, primals, cotangents):
            primals, vjp_f = vjp(f, primals)
            cotangents = vjp_f((cotangents))[0]
            return primals, cotangents

        
        start = time.perf_counter()

        vjp_m_compiled = jit(vjp_wrapper, static_argnames=['f']).lower(m, z, Dz).compile()

        end = time.perf_counter()
        total_compilation_time += (end - start)

        del vjp_wrapper, vjp_m_compiled


    return total_compilation_time

def bench_vjp_w_model_jit_compilation(loops, z):

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
runner.bench_time_func('Run VJP', bench_vjp, z, Dz, inner_loops=1)
runner.bench_time_func('Run VJP (w/ JIT)', bench_vjp_w_jit, z, Dz, inner_loops=1)
runner.bench_time_func('Run VJP (model marked for JIT)', bench_vjp_w_model_jit, z, Dz, inner_loops=1)
runner.bench_time_func('Compilation VJP (w/ JIT)', bench_vjp_w_jit_compilation, z, Dz, inner_loops=1)
runner.bench_time_func('Compilation VJP (model marked for JIT)', bench_vjp_w_model_jit_compilation, z, inner_loops=1)