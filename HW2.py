import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import linalg

results = loadmat('yalefaces.mat')
X = results['X']

# Question A
# Make C a matrix of 100x100
# we want the full 32x32 faces but only the first 100 of them
# another matrix derived from x as 1024x2414
# take dot product

Xm = X[:, 0:100]
C = np.matmul(Xm.T, Xm)
A1 = C
# print(A1)

# Question B
# A2 is which two images highest correlated, A3 least correlated
score_max = 0
r_max = 0
c_max = 0
for i in range(100):
    for j in range((i + 1), 100):
        if C[i, j] > score_max:
            score_max = C[i, j]
            r_max = i + 1
            c_max = j + 1

A2 = np.asarray([r_max, c_max])
A2 = A2[np.newaxis, :]

score_min = 100000
r_min = 0
c_min = 0
for i in range(100):
    for j in range((i + 1), 100):
        if C[i, j] < score_min:
            score_min = C[i, j]
            r_min = i + 1
            c_min = j + 1

A3 = np.asarray([r_min, c_min])
A3 = A3[np.newaxis, :]

# Question C
Xm = X[:, [0, 312, 511, 4, 2399, 112, 1023, 86, 313, 2004]]
C = np.matmul(Xm.T, Xm)
A4 = C

# Question D
evals, evectors = linalg.eig(np.matmul(X, X.T))
print(evals, evectors)
A5 = np.absolute(np.array([evectors[:, 0], evectors[:, 1], evectors[:, 2], evectors[:, 3], evectors[:, 4], evectors[:, 5]]).T)

# Question E
u, s, v = np.linalg.svd(X)
A6 = np.abs(u[:, 0:6])

# Question F
A7 = abs(np.linalg.norm(abs(A6[1]) - abs(A5[1])))
# print(A7)

A8 = np.empty(6)
summation = np.sum(s)

for i in range(6):
    A8[i] = (s[i] / summation) * 100

# Used to change A8 from being shape (6,) to (1, 6)
A8 = A8[np.newaxis, :]
# print("A8 shape:", A8.shape)
