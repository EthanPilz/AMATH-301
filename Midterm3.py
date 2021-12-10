import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.optimize


def printInfo(answer):
    print("Shape:", answer.shape, "\n", answer)


# Question 1
n = 200
x = np.linspace(0, 2 * np.pi, n)

s1 = np.sin(1 * x)
s1 = s1 / np.sqrt(np.dot(s1, s1))
s2 = np.sin(2 * x)
s2 = s2 / np.sqrt(np.dot(s2, s2))
s3 = np.sin(3 * x)
s3 = s3 / np.sqrt(np.dot(s3, s3))
s4 = np.sin(4 * x)
s4 = s4 / np.sqrt(np.dot(s4, s4))
s5 = np.sin(5 * x)
s5 = s5 / np.sqrt(np.dot(s5, s5))

c0 = np.cos(0 * x)
c0 = c0 / np.sqrt(np.dot(c0, c0))
c1 = np.cos(1 * x)
c1 = c1 / np.sqrt(np.dot(c1, c1))
c2 = np.cos(2 * x)
c2 = c2 / np.sqrt(np.dot(c2, c2))
c3 = np.cos(3 * x)
c3 = c3 / np.sqrt(np.dot(c3, c3))
c4 = np.cos(4 * x)
c4 = c4 / np.sqrt(np.dot(c4, c4))

A1 = np.asarray([s1, s2, s3, s4, s5, c0, c1, c2, c3, c4])
A1 = A1.T
# printInfo(A1)

# plt.plot(x, s1, x, s2, x, s3, x, s4, x, s5, x, c0, x, c1, x, c2, x, c3, x, c4)
# plt.plot(A1)
# plt.plot(x, s5)
# plt.show()

A2 = np.matmul(A1.T, A1)

# Question 1b
E = np.zeros((1, 10))
finalSummation1 = 0
finalSummation2 = 0

for i in range(0, 5):
    aN_dotproduct = np.matmul(np.exp(-2 * (x - 2) ** 2), A1[:, i])
    aN_sumproduct = aN_dotproduct * A1[:, i]
    finalSummation1 = finalSummation1 + aN_sumproduct

for i in range(5, 10):
    bN_dotproduct = np.matmul(np.exp(-2 * (x - 2) ** 2), A1[:, i])
    bN_sumproduct = bN_dotproduct * A1[:, i]
    finalSummation2 = finalSummation2 + bN_sumproduct

totalSummation = finalSummation1 + finalSummation2

A3 = totalSummation
A3 = np.reshape(A3, (1, 200))
printInfo(A3)

for i in range(1, 11):
    E[0, i - 1] = np.linalg.norm(np.exp(-2 * (x - 2) ** 2) - totalSummation)

A4 = E[0, 0]
# printInfo(A4)

# Question 2
y = np.array((75, 77, 76, 73, 69, 68, 63, 59, 57, 55, 54, 52, 50, 50, 49, 49, 49, 50, 54, 56, 59, 63, 67, 72))
t = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24))

initials = np.array((13, (2 * np.pi) / 24, 1, 63))

# plt.plot(t, y, '-0')
# plt.show()


def sine_leastsquares_error(coeffs):
    A = coeffs[0]
    B = coeffs[1]
    C = coeffs[2]
    D = coeffs[3]

    return np.sqrt(np.sum((A * np.sin((B * t) + C) + D - y) ** 2))


coefficients = scipy.optimize.fmin(sine_leastsquares_error, initials)

A = coefficients[0]
B = coefficients[1]
C = coefficients[2]
D = coefficients[3]

yp2 = A * np.sin((B * t) + C) + D

summation = 0
for i in range(0, 24):
    summation += (y[i] - yp2[i]) ** 2

A5 = np.sqrt(summation / 24)  # Error
# printInfo(A5)

x = np.linspace(1, 24, 2301)
A6 = A * np.sin((B * x) + C) + D
A6 = np.reshape(A6, (2301, 1))
# printInfo(A6)

# Question 2
t = np.array((-3.0, -2.2, -1.7, -1.5, -1.3, -1.0, -0.7, -0.4, -0.25, -0.05, 0.07, 0.15, 0.3, 0.65, 1.1, 1.25, 1.8, 2.5))
y = np.array((-0.2, 0.1, 0.05, 0.2, 0.4, 1, 1.2, 1.4, 1.8, 2.2, 2.1, 1.6, 1.5, 1.1, 0.8, 0.3, -0.1, 0.2))

initials = np.array((0.2, (2 * np.pi) / 5, 2.2, 1.3))

# plt.plot(t, y, '-0')
# plt.show()


def sine_exponential_leastsquares_error(coeffs):
    A = coeffs[0]
    B = coeffs[1]
    C = coeffs[2]
    D = coeffs[3]

    return np.sqrt(np.sum((A * np.sin(B * t) + (C * np.exp(-D * (t ** 2))) - y) ** 2))


coefficients = scipy.optimize.fmin(sine_exponential_leastsquares_error, initials)

A = coefficients[0]
B = coefficients[1]
C = coefficients[2]
D = coefficients[3]

yp3 = A * np.sin(B * t) + (C * np.exp(-D * (t ** 2)))

summation = 0
for i in range(0, 18):
    summation += (y[i] - yp3[i]) ** 2

A7 = np.sqrt(summation / 18)  # Error
# printInfo(A7)

x = np.linspace(-3, 3, 61)
A8 = A * np.sin(B * x) + (C * np.exp(-D * (x ** 2)))
A8 = np.reshape(A8, (61, 1))
# printInfo(A8)

# plt.plot(x, A8[:, 0], '-0')
# plt.show()