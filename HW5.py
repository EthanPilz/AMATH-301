import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Question A1
dt = 0.5
T = 30
tSpan = np.arange(0, 30 + dt, dt)
#          y(0) | y'(0)
initCond = [0.1, -1]


def vanDerPolSolve(y0, eps):
    y1, y2 = y0
    return [y2, -eps * ((y1 ** 2) - 1) * y2 - y1]


A1 = integrate.odeint(lambda t, x: vanDerPolSolve(t, 0.1), initCond, tSpan)
print(A1)

# A2

A2 = integrate.odeint(lambda t, x: vanDerPolSolve(t, 1), initCond, tSpan)
# print(A2)

# A3

# eps = 20
A3 = integrate.odeint(lambda t, x: vanDerPolSolve(t, 20), initCond, tSpan)
# print(A3)

# Question b
trajectory1 = integrate.odeint(lambda t, x: vanDerPolSolve(t, 0.1), initCond, np.arange(0, 10 + dt, dt), atol=1e-11, rtol=1e-11)

trajectory2 = integrate.odeint(lambda t, x: vanDerPolSolve(t, 1), trajectory1[20, :], np.arange(10, 20 + dt, dt), atol=1e-11, rtol=1e-11)

trajectory3 = integrate.odeint(lambda t, x: vanDerPolSolve(t, 20), trajectory2[20, :], np.arange(20, 30 + dt, dt), atol=1e-11, rtol=1e-11)

temp1 = trajectory1[0:-1]
temp2 = trajectory2[0:-1]
A4 = np.vstack((temp1, temp2, trajectory3))

print(A4.shape)
