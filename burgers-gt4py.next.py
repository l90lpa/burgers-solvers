import numpy as np
from scipy import pi
import gt4py.next as gtx
from gt4py.next import float64, neighbor_sum
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time

VertexDim = gtx.Dimension("Vertex")
NeighborDim = gtx.Dimension("Neighbor", kind=gtx.DimensionKind.LOCAL)

V2V = gtx.FieldOffset("V2V", source=VertexDim, target=(VertexDim, NeighborDim))

###### Spatial Operators

# Second-order centered difference approximation of, u_xx, with periodic BCs
@gtx.field_operator
def L(u : gtx.Field[[VertexDim], float64], dx : float64) -> gtx.Field[[VertexDim], float64]:
    return (neighbor_sum(u(V2V), axis=NeighborDim) - 2.0 * u) / (dx * dx)

# Second-order centered difference approximation of, -u * u_x, with periodic BCs
@gtx.field_operator
def N(u : gtx.Field[[VertexDim], float64], dx : float64) -> gtx.Field[[VertexDim], float64]:
    return (u(V2V[1]) - u(V2V[0])) / (2.0 * dx)

# f(u) = nu * u_xx - u * u_x, hence burger's equation is, u_t = f(u)
@gtx.field_operator
def f(u : gtx.Field[[VertexDim], float64], dx : float64, nu : float64) -> gtx.Field[[VertexDim], float64]:
    return nu * L(u, dx) - u * N(u, dx)

###### Time Intergration Method

@gtx.field_operator
def RK4_stage1(y : gtx.Field[[VertexDim], float64], dt : float64, dx : float64, nu : float64) -> gtx.Field[[VertexDim], float64]:
    return dt * f(y, dx, nu)

@gtx.field_operator
def RK4_stage2(y : gtx.Field[[VertexDim], float64], dt : float64, dx : float64, nu : float64, k1 : gtx.Field[[VertexDim], float64]) -> gtx.Field[[VertexDim], float64]:
    return dt * f(y + 0.5 * k1, dx, nu)

@gtx.field_operator
def RK4_stage3(y : gtx.Field[[VertexDim], float64], dt : float64, dx : float64, nu : float64, k2 : gtx.Field[[VertexDim], float64]) -> gtx.Field[[VertexDim], float64]:
    return dt * f(y + 0.5 * k2, dx, nu)

@gtx.field_operator
def RK4_stage4(y : gtx.Field[[VertexDim], float64], dt : float64, dx : float64, nu : float64, k3 : gtx.Field[[VertexDim], float64]) -> gtx.Field[[VertexDim], float64]:
    return dt * f(y + k3, dx, nu)

@gtx.field_operator
def RK4_accumulate(y : gtx.Field[[VertexDim], float64], 
                   k1 : gtx.Field[[VertexDim], float64], 
                   k2 : gtx.Field[[VertexDim], float64], 
                   k3 : gtx.Field[[VertexDim], float64], 
                   k4 : gtx.Field[[VertexDim], float64]) -> gtx.Field[[VertexDim], float64]:
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

@gtx.program
def apply_RK4_step2(y : gtx.Field[[VertexDim], float64], 
                    k1 : gtx.Field[[VertexDim], float64], 
                    k2 : gtx.Field[[VertexDim], float64], 
                    k3 : gtx.Field[[VertexDim], float64], 
                    k4 : gtx.Field[[VertexDim], float64], 
                    dt : float64, 
                    dx : float64, 
                    nu : float64, 
                    out : gtx.Field[[VertexDim], float64]):
    RK4_stage1(y, dt, dx, nu, out=k1)
    RK4_stage2(y, dt, dx, nu, k1, out=k2)
    RK4_stage3(y, dt, dx, nu, k2, out=k3)
    RK4_stage4(y, dt, dx, nu, k3, out=k4)
    RK4_accumulate(y, k1, k2, k3, k4, out=out)

def RK4(y0, dt, num_steps, dx, nu, V2V_offset_provider):
    y = y0

    y_next = gtx.np_as_located_field(VertexDim)(np.empty_like(y0))
    k1 = gtx.np_as_located_field(VertexDim)(np.empty_like(y0))
    k2 = gtx.np_as_located_field(VertexDim)(np.empty_like(y0))
    k3 = gtx.np_as_located_field(VertexDim)(np.empty_like(y0))
    k4 = gtx.np_as_located_field(VertexDim)(np.empty_like(y0))
    print(f"Integrating...")
    for i in tqdm(range(num_steps)):
        # tic = time.perf_counter()
        
        apply_RK4_step2(y, k1, k2 ,k3, k4, dt, dx, nu, y_next, offset_provider={"V2V": V2V_offset_provider})

        # toc = time.perf_counter()
        # print(f"apply_RK4_step at step {i} in {toc - tic:0.4f} seconds")

        y = y_next

    return y

###### Solver

def solver(nu, u_initial, dx, num_points, dt, num_steps):


    vertex_to_vertex_table = np.empty((num_points,2));
    vertex_to_vertex_table[0] = [num_points - 1, 1]
    for i in range(1, num_points - 1):
        vertex_to_vertex_table[i] = [i - 1, i + 1]
    vertex_to_vertex_table[num_points - 1] = [num_points - 2, 0]
    V2V_offset_provider = gtx.NeighborTableOffsetProvider(vertex_to_vertex_table, VertexDim, VertexDim, 2)

    u0 = gtx.np_as_located_field(VertexDim)(u_initial)

    u_end = RK4(u0, dt, num_steps, dx, nu, V2V_offset_provider)
    
    return np.asarray(u_end)

###### Usage

if __name__ == "__main__":
    # Define problem parameters
    num_points = 100
    domain_length = 2 * pi
    dx = domain_length / (num_points - 1)
    nu = 0.1
    dt = 0.001  # Time step size

    # Create the initial velocity profile (e.g., a sinusoidal profile)
    x = np.linspace(0, domain_length, num_points)
    u_initial = np.sin(x)

    # Solve the Burgers' equation
    num_steps = 3140
    u_final = solver(nu, u_initial, dx, num_points, dt, num_steps)

    # Plot the results
    plt.plot(x, u_initial, label="Initial Velocity Profile")
    plt.plot(x, u_final, label=f"Velocity Profile after {num_steps} time steps")
    plt.xlabel("Position (x)")
    plt.ylabel("Velocity (u)")
    plt.legend()
    plt.show()