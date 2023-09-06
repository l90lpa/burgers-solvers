from jax import jit
import jax.numpy as jnp

# If pred == True then `1` else if pred == False then `-1`
def one_or_minusone(pred):
    return (2 * pred) - 1

# If pred == True then `a`, else if pred == False then `b`
def convex_combination(pred, a, b):
    return (pred *  a) + ((1-pred) * b)

@jit
def ppmlin_step_advection(z,v,dt,dx):

    # subroutine for Lin scheme in 1D

    # flux to left 

    deltaz = 0.25 * (jnp.roll(z, -1) - jnp.roll(z, 1))

    deltazmin = z - jnp.minimum(jnp.minimum(jnp.roll(z, 1), jnp.roll(z, -1)), z)
    deltazmax = jnp.maximum(jnp.maximum(jnp.roll(z, 1), jnp.roll(z, -1)), z) - z
  
    minpart = jnp.minimum(jnp.minimum(deltazmin, deltazmax), jnp.abs(deltaz))
    deltazmono = one_or_minusone(jnp.greater_equal(deltaz, 0.0)) * abs(minpart)
                
    
    # Approx of z at the cell edge

    zminus = 0.5 * (jnp.roll(z, 1) + z) + (1.0/3.0) * (jnp.roll(deltazmono, 1) - deltazmono)
    
    # continuous at cell edges

    zplus = jnp.roll(zminus, -1)
    
    # limit cell edges
          
    dl = jnp.minimum(2.0 * jnp.abs(deltazmono), jnp.abs(zminus - z))
    dr = jnp.minimum(2.0 * jnp.abs(deltazmono), jnp.abs(zplus - z))
    dl = one_or_minusone(jnp.greater_equal(deltazmono, 0.0)) * jnp.abs(dl)
    dr = one_or_minusone(jnp.greater_equal(deltazmono, 0.0)) * jnp.abs(dr)    
    zminusnew = z - dl
    zplusnew = z + dr
    


    zminus = zminusnew
    zplus = zplusnew
    z6 = 3.0 * (dl - dr)
        
    # calculate flux based on velocity direction

    pred = (v >= 0.0)
    flux = convex_combination(pred, jnp.roll(zplus, 1), zminus) - one_or_minusone(pred) * 0.5*(abs(v)*dt/dx)*( convex_combination(pred, jnp.roll(zplus - zminus, 1), zplus - zminus) - one_or_minusone(pred) * convex_combination(pred, jnp.roll(z6, 1), z6)*(1.0-(2.0/3.0)*(abs(v)*dt/dx)) )

    z_new = -(1.0/dx) * (jnp.roll(flux, -1) - flux)

    return z_new

