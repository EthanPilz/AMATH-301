import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


def printInfo(answer):
    print(answer.shape, "\n", answer)


# Question A
n = 200
x = np.linspace(0, 2 * np.pi, n)

s1 = np.sin(1 * x / 2)
s1 = s1 / np.sqrt(np.dot(s1, s1))
s2 = np.sin(2 * x / 2)
s2 = s2 / np.sqrt(np.dot(s2, s2))
s3 = np.sin(3 * x / 2)
s3 = s3 / np.sqrt(np.dot(s3, s3))
s4 = np.sin(4 * x / 2)
s4 = s4 / np.sqrt(np.dot(s4, s4))
s5 = np.sin(5 * x / 2)
s5 = s5 / np.sqrt(np.dot(s5, s5))
s6 = np.sin(6 * x / 2)
s6 = s6 / np.sqrt(np.dot(s6, s6))
s7 = np.sin(7 * x / 2)
s7 = s7 / np.sqrt(np.dot(s7, s7))
s8 = np.sin(8 * x / 2)
s8 = s8 / np.sqrt(np.dot(s8, s8))
s9 = np.sin(9 * x / 2)
s9 = s9 / np.sqrt(np.dot(s9, s9))
s10 = np.sin(10 * x / 2)
s10 = s10 / np.sqrt(np.dot(s10, s10))

A1 = np.asarray([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10])
A1 = A1.T
# print(A1.shape)
# print(A1)

# plt.plot(x, s1, x, s2, x, s3, x, s4, x, s5, x, s6, x, s7, x, s8, x, s9, x, s10)
# plt.plot(A1)
# plt.show()

A2 = np.matmul(A1.T, A1)
# print(A2.shape)

# Question b
n = 200
x = np.linspace(0, 2 * np.pi, n)

s1 = np.cos(0 * x / 2)
s1 = s1 / np.sqrt(np.dot(s1, s1))
s2 = np.cos(1 * x / 2)
s2 = s2 / np.sqrt(np.dot(s2, s2))
s3 = np.cos(2 * x / 2)
s3 = s3 / np.sqrt(np.dot(s3, s3))
s4 = np.cos(3 * x / 2)
s4 = s4 / np.sqrt(np.dot(s4, s4))
s5 = np.cos(4 * x / 2)
s5 = s5 / np.sqrt(np.dot(s5, s5))
s6 = np.cos(5 * x / 2)
s6 = s6 / np.sqrt(np.dot(s6, s6))
s7 = np.cos(6 * x / 2)
s7 = s7 / np.sqrt(np.dot(s7, s7))
s8 = np.cos(7 * x / 2)
s8 = s8 / np.sqrt(np.dot(s8, s8))
s9 = np.cos(8 * x / 2)
s9 = s9 / np.sqrt(np.dot(s9, s9))
s10 = np.cos(9 * x / 2)
s10 = s10 / np.sqrt(np.dot(s10, s10))

A3 = np.asarray([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10])
A3 = A3.T
# printInfo(A3)

A4 = np.matmul(A3.T, A3)
# printInfo(A4)
# Question c
A5 = np.matmul(A1.T, A3)
# printInfo(A5)

# Question d
E = np.zeros((1, 10))
finalSummation = 0

for i in range(1, 11):
    dotProduct = np.matmul(np.cos(x), A1[:, i - 1])
    sumProduct = dotProduct * A1[:, i - 1]
    finalSummation = finalSummation + sumProduct
    E[0, i - 1] = np.linalg.norm(np.cos(x) - finalSummation)

# printInfo(sumProduct)

printInfo(E)
A6 = E
