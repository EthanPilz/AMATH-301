import numpy as np
from scipy import linalg

# Question 1a - Jacobi Iterations
from scipy.io import loadmat

A1 = np.zeros((6, 3))

x = np.array([0])
y = np.array([0])
z = np.array([0])

for j in range(6):
    x = np.append(x, (1 - 2 * y[j] - z[j]) / 8)
    y = np.append(y, (2 + x[j] - z[j]) / 5)
    z = np.append(z, (-x[j] - y[j]) / 4)

    if abs(x[j + 1] - x[j]) < 1e-6 and abs(y[j + 1] - x[j]) < 1e-6 and abs(z[j + 1] - x[j]) < 1e-6:
        break

    A1[j, 0] = x[j + 1]
    A1[j, 1] = y[j + 1]
    A1[j, 2] = z[j + 1]

A1 = A1.T
print(A1)
# print("A1:\n", A1)
# print("A1 shape:", A1.shape)

# Question 1b - Gauss-Seidel Iterations

A2 = np.zeros((6, 3))

x = np.array([0])
y = np.array([0])
z = np.array([0])

for j in range(6):
    x = np.append(x, (1 - 2 * y[j] - z[j]) / 8)
    y = np.append(y, (2 + x[j + 1] - z[j]) / 5)
    z = np.append(z, (-x[j + 1] - y[j + 1]) / 4)

    if abs(x[j + 1] - x[j]) < 1e-6 and abs(y[j + 1] - x[j]) < 1e-6 and abs(z[j + 1] - x[j]) < 1e-6:
        break

    A2[j, 0] = x[j + 1]
    A2[j, 1] = y[j + 1]
    A2[j, 2] = z[j + 1]

A2 = A2.T
# print("A2:\n", A2)
# print("A2 shape:", A2.shape)

# Question 1c - Jacobi Iterations
A3 = np.zeros((6, 3))

x = np.array([0])
y = np.array([0])
z = np.array([0])

for j in range(6):
    x = np.append(x, (1 - y[j] - 4 * z[j]))
    y = np.append(y, (2 + x[j] - z[j]) / 5)
    z = np.append(z, (-8 * x[j] - 2 * y[j]))

    if abs(x[j + 1] - x[j]) < 1e-6 and abs(y[j + 1] - x[j]) < 1e-6 and abs(z[j + 1] - x[j]) < 1e-6:
        break

    A3[j, 0] = x[j + 1]
    A3[j, 1] = y[j + 1]
    A3[j, 2] = z[j + 1]

A3 = A3.T
# print("A3:\n", A3)
# print("A3 shape:", A3.shape)

# Question 1d - Gauss-Seidel Iterations
A4 = np.zeros((6, 3))

x = np.array([0])
y = np.array([0])
z = np.array([0])

for j in range(6):
    x = np.append(x, (1 - y[j] - 4 * z[j]))
    y = np.append(y, (2 + x[j + 1] - z[j]) / 5)
    z = np.append(z, (-8 * x[j + 1] - 2 * y[j + 1]))

    if abs(x[j + 1] - x[j]) < 1e-6 and abs(y[j + 1] - x[j]) < 1e-6 and abs(z[j + 1] - x[j]) < 1e-6:
        break

    A4[j, 0] = x[j + 1]
    A4[j, 1] = y[j + 1]
    A4[j, 2] = z[j + 1]

A4 = A4.T
# print("A4:\n", A4)
# print("A4 shape:", A4.shape)

# Question 2a - Correlation Matrix

results = loadmat('yalefaces.mat')
X = results['X']

Xm = X[:, [36, 531, 1712]]
C = np.matmul(Xm.T, Xm)
A5 = C

# print("A5:\n", A5)
# print("A5 shape:", A5.shape)

# Question 2b - Eigenvector/value sorting

evalues, evectors = linalg.eig(C)
evectors = np.abs(evectors)
A6 = np.array([[evalues[1], evalues[2], evalues[0]]])

# print("A6:\n", A6)
# print("A6 shape:", A6.shape)

# print("evecs\n", evectors)
# evectors_sorted = np.array([[evectors[1, 1], evectors[0, 0], evectors[0, 2]],
# [evectors[2, 1], evectors[2, 0], evectors[1, 2]],
# [evectors[2, 2], evectors[1, 0], evectors[0, 1]]])

sorted_indices = np.argsort(evalues)
evectors_sorted = evectors[:, sorted_indices]
A7 = evectors_sorted
# print("\n evecs sorted \n", evectors_sorted)

# Question 2c - projection
u, s, v = np.linalg.svd(X)

# Taking the entire column of the image, dot product with 5th column of U.
# take absolute value of that, set it equal to A8. Shape (1, 3).

A8 = np.zeros((1, 3))
#A8 = np.array([[(np.inner(X[:, 36], u[:, 5])),
               # (np.inner(X[:, 531], u[:, 5])),
                #(np.inner(X[:, 1712], u[:, 5]))]])

A8[0, 0] = np.matmul(u[:, 4], X[:, 36])
A8[0, 1] = np.matmul(u[:, 4], X[:, 531])
A8[0, 2] = np.matmul(u[:, 4], X[:, 1712])

A8 = np.abs(A8)

# temp1 = X[:, [36, 531, 1712]]
# A8 = np.abs(np.matmul(temp1.T, u[:, 5]))
print("\n A8: \n", A8)
print(A8.shape)
