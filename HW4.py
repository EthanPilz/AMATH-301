import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import linalg

# Question a
n = 400
h = 0.01
x = np.arange(0, 4 + 0.01, 0.01)  # start, end, step size
y = np.exp(-0.1 * x) * np.cos(5 * x) + ((x ** 2) - 0.1 * (x ** 4))  # function to integrate
yi_exact = np.exp(-0.1 * x) * np.cos(x) + ((x ** 2) - 0.1 * x)  # y_integrate exact

# Trapezoidal rule
TotalAreaT = np.zeros(n + 1)
TotalAreaT[0] = 0
for j in range(0, n):
    A = h * (y[j] + y[j + 1]) / 2
    TotalAreaT[j + 1] = A + TotalAreaT[j]

# print(TotalAreaT, "\n", TotalAreaT.shape, "\n", "\n")
A1 = TotalAreaT
A1 = A1[np.newaxis, :]
print(A1)
#print(A1.shape)


# Simpson's rule
m = 200
TotalAreaS = np.zeros(m + 1)
TotalAreaS[0] = 0
count = 0
for j in range(0, n - 1, 2):
    A = h * (y[j] + 4 * y[j + 1] + y[j + 2]) / 3
    TotalAreaS[count + 1] = A + TotalAreaS[count]
    count = count + 1

# print(TotalAreaS)
# print(TotalAreaS.shape)
A2 = TotalAreaS
#print(A2)
A2 = A2[np.newaxis, :]

# Question b
dx = 0.01
x = np.arange(0, 4 + dx, dx)  # start, end, step size
y = np.exp(-0.1 * x) * np.cos(5 * x) + ((x ** 2) - 0.1 * (x ** 4))  # function to derivate
n = np.size(x)

yp4 = np.zeros(n)
yp4[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * dx)
for j in range(1, n - 1):
    yp4[j] = (y[j + 1] - y[j - 1]) / (2 * dx)

yp4[n - 1] = (3 * y[n - 1] - 4 * y[n - 2] + y[n - 3]) / (2 * dx)

A3 = yp4
A3 = A3[np.newaxis, :]
# print(A3, "\n \n \n")

# Question c
dx = 0.01
x = np.arange(0, 4 + dx, dx)  # start, end, step size
y = np.exp(-0.1 * x) * np.cos(5 * x) + ((x ** 2) - 0.1 * (x ** 4))  # function to derivate
n = np.size(x)

yp4 = np.zeros(n)
yp4[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * dx)
yp4[1] = (-3 * y[1] + 4 * y[2] - y[3]) / (2 * dx)
for j in range(2, n - 2):
    yp4[j] = (-y[j + (2 * 1)] + (8 * y[j + 1]) - (8 * y[j - 1]) + y[j - (2 * 1)]) / (12 * dx)

# the last two endpoints were positive, put a negative sign to fix
yp4[n - 1] = -(-3 * y[n - 1] + 4 * y[n - 2] - y[n - 3]) / (2 * dx)
yp4[n - 2] = -(-3 * y[n - 2] + 4 * y[n - 3] - y[n - 4]) / (2 * dx)

A4 = yp4
A4 = A4[np.newaxis, :]
# print(A4)
# plt.plot(x, y)
# plt.plot(x, yp4, ':')
# plt.show()

# Question d

dx = 0.01
x = np.arange(0, 4 + dx, dx)  # start, end, step size
y = np.exp(-0.1 * x) * np.cos(5 * x) + ((x ** 2) - 0.1 * (x ** 4))  # function to derivate
n = np.size(x)

yp5 = np.zeros(n)
yp5[0] = (-3 * yp4[0] + 4 * yp4[1] - yp4[2]) / (2 * dx)
yp5[1] = (-3 * yp4[1] + 4 * yp4[2] - yp4[3]) / (2 * dx)
for j in range(2, n - 2):
    yp5[j] = (-yp4[j + (2 * 1)] + (8 * yp4[j + 1]) - (8 * yp4[j - 1]) + yp4[j - (2 * 1)]) / (12 * dx)

# the last two endpoints were positive, put a negative sign to fix?
yp5[n - 1] = -(-3 * yp4[n - 1] + 4 * yp4[n - 2] - yp4[n - 3]) / (2 * dx)
yp5[n - 2] = -(-3 * yp4[n - 2] + 4 * yp4[n - 3] - yp4[n - 4]) / (2 * dx)
A5 = yp5
#plt.plot(x, y)
#plt.plot(x, yp5, ':')
#plt.show()
A5 = A5[np.newaxis, :]
# print(A5)

# Question e
dx = 0.01
x = np.arange(0, 4 + dx, dx)  # start, end, step size
y = np.exp(-0.1 * x) * np.cos(5 * x) + ((x ** 2) - 0.1 * (x ** 4))  # function to integrate
n = np.size(x)

yp6 = np.zeros(n)
yp6[0] = ((2 * y[0]) - (5 * y[0 + 1]) + (4 * y[0 + (2 * 1)]) - (y[0 + (3 * 1)])) / (dx ** 2)
yp6[1] = ((2 * y[1]) - (5 * y[1 + 1]) + (4 * y[1 + (2 * 1)]) - (y[1 + (3 * 1)])) / (dx ** 2)
for j in range(2, n - 2):
    yp6[j] = (-y[j + (2 * 1)] + (16 * y[j + 1]) - (30 * y[j]) + (16 * y[j - 1]) - y[j - (2 * 1)]) / (12 * (dx ** 2))

# the last two endpoints were positive, put a negative sign to fix
yp6[n - 1] = ((2 * y[n - 1]) - (5 * y[(n - 1) - 1]) + (4 * y[(n - 1) - (2 * 1)]) - (y[(n - 1) - (3 * 1)])) / (dx ** 2)
yp6[n - 2] = ((2 * y[n - 2]) - (5 * y[(n - 2) - 1]) + (4 * y[(n - 2) - (2 * 1)]) - (y[(n - 2) - (3 * 1)])) / (dx ** 2)
plt.plot(x, y)
plt.plot(x, yp6, ':')
plt.show()
A6 = yp6
A6 = A6[np.newaxis, :]
#print(A6)

# Question f

A7 = np.linalg.norm(A6 - A5)
#print("\n A7 :", A7)