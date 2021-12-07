import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import CubicSpline
import scipy.optimize


def printInfo(answer):
    print("Shape:", answer.shape, "\n", answer)


# Question 1 A
t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
y = [75, 77, 76, 73, 69, 68, 63, 59, 57, 55, 54, 52, 50, 50, 49, 49, 49, 50, 54, 56, 59, 63, 67, 72]

pcoeff = np.polyfit(t, y, 2)
xp = np.arange(1, 25, 1)
yp = np.polyval(pcoeff, xp)

summation = 0
for i in range(0, 24):
    summation += (y[i] - yp[i]) ** 2

A1 = np.sqrt(summation / 24)
A2 = np.polyval(pcoeff, np.arange(1, 24.01, 0.01))
A2 = np.reshape(A2, (2301, 1))
# printInfo(A2)

# Question 1 B

A3 = np.interp(np.linspace(1, 24, 2301), t, y)
A3 = np.reshape(A3, (2301, 1))
# printInfo(A3)

spl = CubicSpline(t, y)
A4 = spl(np.linspace(1, 24, 2301))
A4 = np.reshape(A4, (2301, 1))

# printInfo(A4)

# Question C
y = np.array((75, 77, 76, 73, 69, 68, 63, 59, 57, 55, 54, 52, 50, 50, 49, 49, 49, 50, 54, 56, 59, 63, 67, 72))
t = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24))
initials = np.array((12, (2 * np.pi) / 24, 60))


def cosine_leastsquares_error(coeffs):
    A = coeffs[0]
    B = coeffs[1]
    C = coeffs[2]

    return np.sqrt(np.sum((A * np.cos(B * t) + C - y) ** 2))


coefficients = scipy.optimize.fmin(cosine_leastsquares_error, initials)

A = coefficients[0]
B = coefficients[1]
C = coefficients[2]

print(A, B, C)

yp2 = A * np.cos(B * t) + C

summation = 0
for i in range(0, 24):
    summation += (y[i] - yp2[i]) ** 2

A5 = np.sqrt(summation / 24)
printInfo(A5)

x = np.linspace(1, 24, 2301)
A6 = A * np.cos(B * x) + C
A6 = np.reshape(A6, (2301, 1))
# printInfo(A6)

# Question 2

s = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
v = np.array(
    [30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55,
     54, 53])
initials = np.array((3, np.pi / 4, 2 / 3, 32))


def cos_leastSquare_2(coeffs):
    A = coeffs[0]
    B = coeffs[1]
    C = coeffs[2]
    D = coeffs[3]

    return np.sqrt(np.sum((A * np.cos(B * s) + (C * s) + D - v) ** 2))


coefficients2 = scipy.optimize.fmin(cos_leastSquare_2, initials)

A = coefficients2[0]
B = coefficients2[1]
C = coefficients2[2]
D = coefficients2[3]

x = np.arange(0, 30.01, 0.01)
A7 = A * np.cos(B * x) + (C * x) + D
A7 = np.reshape(A7, (1, 3001))
# printInfo(A7)
