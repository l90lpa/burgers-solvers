# PDE Solvers and Automatic Differentiation

This project contains a collection of solvers, various PDEs, for experimenting with automatic differentiation.

## Dependencies

The project has been tested using python version 3.10.12, and the package dependecies are specified in `requirements.txt`.

## Set-up Python environment

### Create a Python virtual environment

Assuming commands are executed on the cmd line at the project root.
- To create a virtual environment: ```python -m venv .venv```
- To activate the virtual environment: ```source ./.venv/bin/activate```

### Install packages using pip

- In a terminal at the project root: ```pip install -r requirements.txt```

## Overview of the Repo

- solvers-jax: contains PDE in solvers written using JAX, as well as, in most cases a tangent linear model (TLM) and an adjoint model (ADM) of the solver.
    - advection-*-jax.py: a linear advecttion solver, and its corresponding TLM and ADM, written in JAX. These scripts can be executed as follows `python advection-*-jax.py` where one replaces the `*` with the appropriate string.
    - advection-rk3-mpi4jax.py: a linear advection solver, and its corresponding TLM and ADM, written using JAX, and mpi4jax to enable distributed computation. The script can be executed as follows `mpirun -np 4 python advection-rk3-mpi4jax.py` where one replaces the `*` with the appropriate string.
    - burgers-jax.py: a viscid Burger's equation solver written using JAX. The script can be executed as follows: `python burgers-jax.py`
    - ab3.py, rk3.py, and ppm.py: contain time integration routines, and there associated TLM and ADM.

- benchmarks-jax: contains a collection of benchmarks showcasing various combinations of the JAX transforms, `jit`, `jvp`, and `vjp`. Each script can be executed as follows `python script.py`
