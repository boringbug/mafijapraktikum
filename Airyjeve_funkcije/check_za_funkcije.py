from definicije import Ai_taylor, Ai_neg_asim, Ai_pos_asim
from scipy.special import gamma
from numpy import append, pow, linspace, sqrt, exp, abs, pi, sin, cos, append, matrix, max
import matplotlib.pyplot as plt

from scipy.special import airy

def u(s):
    return gamma(3*s + 1/2)/(pow(54, s)*gamma(s + 1/2)*gamma(s+1))

def u_better(s):
    return (3*s + 5/2)*(3*s + 1/2)/(18*(s+1))

def L_real(z, n):
    sum = 0
    for s in range(0, n):
        sum += u(s)/pow(z, s)
    return sum

def P_real(z, n):
    sum = 0
    for s in range(0, n):
        sum += pow(-1, s)*u(2*s)/pow(z, 2*s)
    return sum

def Q_real(z, n):
    sum = 0
    for s in range(0, n):
        sum += pow(-1, s)*u(2*s+1)/pow(z, 2*s+1)
    return sum


def Ai_neg_asim_real(x, n):
    ksi = 2/3 * pow(abs(x), 3/2)
    return (sin(ksi - pi/4)*Q_real(ksi, n) + cos(ksi - pi/4)*P_real(ksi, n))/(sqrt(pi)*pow(-x, 1/4))


x = linspace(-100, 100, 10000)
x_pos = linspace(7, 100, 10000)
x_neg = linspace(-100, -8, 10000)

y = airy(x)[0]
y_pos = airy(x_pos)[0]
y_neg = airy(x_neg)[0]


diff_taylor = matrix([abs(y - Ai_taylor(x, n)) for n in range(100, 110)])
print("Taylor: ")
print(diff_taylor)
print("Max diff of taylor: " + str(diff_taylor.max()))

diff_pos = matrix([abs(y_pos - Ai_pos_asim(x_pos, n)) for n in range(50, 60)])
print("Pos: ")
print(diff_pos)
print("Max diff of pos: " + str(diff_pos.max()))

diff_neg = matrix([abs(y_neg - Ai_neg_asim(x_neg, n)) for n in range(50, 60)])
print("Neg: ")
print(diff_neg)
print("Max diff of neg: " + str(diff_neg.max()))

plt.plot(x, Ai_taylor(x, 50))
plt.plot(x, Ai_pos_asim(x, 50))
plt.plot(x, Ai_neg_asim(x, 50))
plt.plot(x, y)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.legend()
plt.show()
