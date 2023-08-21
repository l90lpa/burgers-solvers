import numpy as np
from scipy import sparse

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

def create_diffusion_matrix(dx, n):

    # Centered second-order finite difference approximation of u_xx on a periodic domain
    dxdx = (dx ** 2)
    vals = np.array([-2.0, 1.0, 1.0]) / dxdx
    offsets = np.array([0, 1, n-1])
    dupvals = np.concatenate((vals, vals[::-1]))
    dupoffsets = np.concatenate((offsets, -offsets))
    A = sparse.diags(dupvals, dupoffsets, shape=(n, n)).tocsr()

    return A

def create_fourth_order_diffusion_operator(dx, n, nu2):

    # Centered second-order finite difference approximation of u_xxxx on a periodic domain
    def diffusion2_op(u):
        u_new = np.empty_like(u)
        u_new[0] = -nu2 * (u[n-2] - 4 * u[n-1] + 6 * u[0] - 4 * u[1] + u[2]) / (dx ** 4)
        u_new[1] = -nu2 * (u[n-1] - 4 * u[0] + 6 * u[1] - 4 * u[2] + u[3]) / (dx ** 4)
        for i in range(2, n-2):
            u_new[i] = -nu2 * (u[i-2] - 4 * u[i-1] + 6 * u[i] - 4 * u[i+1] + u[i+2]) / (dx ** 4)
        u_new[n-2] = -nu2 * (u[n-4] - 4 * u[n-3] + 6 * u[n-2] - 4 * u[n-1] + u[0]) / (dx ** 4)
        u_new[n-1] = -nu2 * (u[n-3] - 4 * u[n-2] + 6 * u[n-1] - 4 * u[0] + u[1]) / (dx ** 4)
        return u_new
    
    return diffusion2_op

def create_fourth_order_diffusion_matrix(dx, n, nu2):

    # Centered second-order finite difference approximation of u_xx on a periodic domain
    vals = -(nu2 / (dx ** 4)) * np.array([6.0, -4.0, 1.0, 1.0, -4.0])
    offsets = np.array([0, 1, 2, n-2, n-1])
    dupvals = np.concatenate((vals, vals[:0:-1]))
    dupoffsets = np.concatenate((offsets, -offsets[1:5]))
    A = sparse.diags(dupvals, dupoffsets, shape=(n, n)).tocsr()

    return A

def create_advection_opeartor(dx, n):
    
    # Centered second-order finite difference approximation of u_x on a periodic domain
    def advection_op(u):
        dx2 = dx * 2
        u_new = np.empty_like(u)
        u_new[0] = (u[1] - u[n-1]) / dx2
        for i in range(1, n-1):
            u_new[i] = (u[i+1] - u[i-1]) / dx2
        u_new[n-1] = (u[1] - u[n-1]) / dx2
        return u_new
    
    return advection_op