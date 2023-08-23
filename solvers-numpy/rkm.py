
def RK3_step(f, y, dt):

    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.75 * dt * k2)

    return (1.0 / 9.0) * (2 * k1 + 3 * k2 + 4 * k3)


def RK3(f, y0, dt, num_steps):
    y = y0

    for i in range(num_steps):
        y = y + dt * RK4_step(f, y, dt)

    return y


def RK4_step(f, y, dt):

    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)

    return (1.0 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def RK4(f, y0, dt, num_steps):
    y = y0

    for i in range(num_steps):
        y = y + dt * RK4_step(f, y, dt)

    return y