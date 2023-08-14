import numpy as np

def create_diffusion_opeartor(dx, n):

    # Centered second-order finite difference approximation of u_xx on a periodic domain
    def diffusion_op(u):
        dxdx = dx ** 2
        u_new = np.empty_like(u)
        u_new[0] = (u[n-1] -2 * u[0] + u[1]) / dxdx
        for i in range(1, n-1):
            u_new[i] = (u[i-1] - 2 * u[i] + u[i+1]) / dxdx
        u_new[n-1] = (u[n-2] - 2 * u[n-1] + u[0]) / dxdx
        return u_new
    
    return diffusion_op

def create_advection_opeartor(dx, n):
    
    # Centered second-order finite difference approximation of u_x on a periodic domain
    def advection_op(u):
        dx2 = dx * 2
        u_new = np.empty_like(u)
        u_new[0] = (u[1] - u[n-2]) / dx2
        for i in range(1, n-1):
            u_new[i] = (u[i+1] - u[i-1]) / dx2
        u_new[n-1] = (u[1] - u[n-2]) / dx2
        return u_new
    
    return advection_op