import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
from scipy.special import airy, airye, gamma

# Ai(x) = a f(x) + b g(x) in Bi(x) = sqrt(3)[a f(x) - b g(x)]
# a = Ai(0) = Bi(0)/sqrt(3) in b = Ai'(0) = -Bi'(0)/sqrt(3)

a, b, _, _ = airy(0)

constant_gamma_f = gamma(1/3)
constant_gamma_g = gamma(2/3)

N = 100 # This is the limiting number in the sum

def BIfactor(x, n):
    return gamma(x + n)/gamma(x)

def f(x):
    sum = 1
    product = 1
    for k in range(1, N):
        product *= (k+1/3)/(3*k*constant_gamma_f)
        sum += product * 1/gamma(3*k) * 3**k * x**(3*k)
    return sum

def g(x):
    sum = 0
    product = 1
    for k in range(1, N):
        product *= (k+2/3)/(3*k*constant_gamma_f)
        sum += BIfactor(2/3, k)/constant_gamma_g * 1/gamma(3*k+1) * 3**k * x**(3*k+1)
    return sum + x

def Ai(x):
    return a*f(x) + b*g(x)

def Bi(x):
    return 1/np.sqrt(3) * (a*f(x) - b*g(x))

x = np.linspace(-5, 5, 1000)
Ai_real, _, Bi_real, _ = airy(x)

plt.plot(x, Ai(x), label='Ai vrsta')
plt.plot(x, Ai_real, label='Ai dejanska')
#plt.plot(x, Bi(x), label='Bi vrsta')
#plt.plot(x, Bi_real, label='Bi dejanska')

plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.legend()

plt.grid(True)
plt.show()
