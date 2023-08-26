from jax import config, jit
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from math import pi, ceil
from functools import partial
import time
import numpy as np
import matplotlib.pyplot as plt

# If pred == True then `1` else if pred == False then `-1`
def one_or_minusone(pred):
    return (2 * pred) - 1

# If pred == True then `a`, else if pred == False then `b`
def convex_combination(pred, a, b):
    return (pred *  a) + ((1-pred) * b)

@jit
def ppmlin_step_advection(z,v,dt,dx):

    # subroutine for Lin scheme in 1D
    N = jnp.size(z)
    flux = jnp.empty((N,), dtype=np.float64)
    deltaz = jnp.empty((N,),dtype=np.float64)
    deltazmax = jnp.empty((N,),dtype=np.float64)
    deltazmin = jnp.empty((N,),dtype=np.float64)
    deltazmono = jnp.empty((N,),dtype=np.float64)
    zplus = jnp.empty((N,),dtype=np.float64)
    zminus = jnp.empty((N,),dtype=np.float64)
    dl = jnp.empty((N,),dtype=np.float64)
    dr = jnp.empty((N,),dtype=np.float64)
    zminusnew = jnp.empty((N,),dtype=np.float64)
    zplusnew = jnp.empty((N,),dtype=np.float64)
    z6 = jnp.empty((N,),dtype=np.float64)

    # flux to left 

    for i in range(N):
        ai=(i+1) % N
        bi=(i-1) % N
        
        deltaz = deltaz.at[i].set(0.25*(z[ai] - z[bi]))

        deltazmin = deltazmin.at[i].set(z[i]-jnp.min(jnp.array([z[bi],z[i],z[ai]])))
        deltazmax = deltazmax.at[i].set(jnp.max(jnp.array([z[bi],z[i],z[ai]]))-z[i])

            
    for i in range(N):            
        minpart=jnp.min(jnp.array([abs(deltaz[i]),deltazmin[i],deltazmax[i]]))

        deltazmono = deltazmono.at[i].set(one_or_minusone(deltaz[i] >= 0.0)*abs(minpart))
                
    
    # Approx of z at the cell edge

    for i in range(N):
        bi=(i-1) % N
            
        zminus = zminus.at[i].set(0.5*(z[bi]+z[i])+(1.0/3.0)*(deltazmono[bi]-deltazmono[i]))
    
    
    # continuous at cell edges

    for i in range(N):
        ai=(i+1) % N
            
        zplus = zplus.at[i].set(zminus[ai])
    
    
    # limit cell edges
          
    for i in range(N):
           
        dl = dl.at[i].set(jnp.min(jnp.array([abs(2.0*deltazmono[i]),abs(zminus[i]-z[i])])))
        dr = dr.at[i].set(jnp.min(jnp.array([abs(2.0*deltazmono[i]),abs(zplus[i]-z[i])])))

        dl = dl.at[i].set(one_or_minusone(2.0*deltazmono[i] >= 0.0)*abs(dl[i]))
        dr = dr.at[i].set(one_or_minusone(2.0*deltazmono[i] >= 0.0)*abs(dr[i]))
            
        zminusnew = zminusnew.at[i].set(z[i] - dl[i])
        zplusnew = zplusnew.at[i].set(z[i] + dr[i])
    
    
    for i in range(N):           
        zminus = zminus.at[i].set(zminusnew[i])
        zplus = zplus.at[i].set(zplusnew[i])
    
        z6 = z6.at[i].set(3.0*(dl[i]-dr[i]))
    
    
    # calculate flux based on velocity direction

    pred = (v >= 0.0)
    for i in range(N):
        bi=(i-1) % N
                    
        flux = flux.at[i].set(convex_combination(pred, zplus[bi], zminus[i]) - one_or_minusone(pred) * 0.5*(abs(v)*dt/dx)*( convex_combination(pred, zplus[bi]-zminus[bi], zplus[i]-zminus[i]) - one_or_minusone(pred) * convex_combination(pred, z6[bi], z6[i])*(1.0-(2.0/3.0)*(abs(v)*dt/dx)) ))


    z_new = jnp.empty((N,), dtype=np.float64)
    for i in range(N):
        z_new = z_new.at[i].set(-(1.0/dx) * (flux[(i+1)%N] - flux[i]))

    return z_new


def solver(u_initial, v, dx, num_points, dt, num_steps):
    u = np.copy(u_initial)
    
    for i in range(num_steps):
        u = u + v * dt * ppmlin_step_advection(u,v,dt,dx)

    return u
    

# Example usage
if __name__ == "__main__":
    # Define problem parameters
    num_points = 100
    domain_length = 1.0
    dx = domain_length / num_points
    v = 1.2
    C = 0.1 # Courant number
    dt = (dx / v) * C  # Time step size
    num_steps = ceil(1.0 * (domain_length / (dt * v)))

    # Create the initial condition (e.g., a sinusoidal profile)
    x = jnp.linspace(0, domain_length - dx, num_points)
    u_initial = jnp.sin((2*pi/domain_length) * x)

    solver_partial = lambda u0 : solver(u0, v, dx, num_points, dt, num_steps)
    
    # Run solver
    u_final = solver_partial(u_initial)

    # Plot the results
    plt.plot(x, u_initial, label="Initial Condition")
    plt.plot(x, u_final, label=f"Solution after {num_steps} time steps")
    plt.xlabel("Position (x)")
    plt.ylabel("Solution (u)")
    plt.legend()
    plt.show()



