import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
from scipy.special import airy
from definicije import Ai_taylor, Ai_pos_asim, Ai_neg_asim


x_min = -10
x_max = 10

def Ai(x, N):
    if x > -1.5 and x < 1:
        return Ai_taylor(x, N)
    elif x >= 1:
        return Ai_pos_asim(x, N)
    elif x <= -1:
        return Ai_neg_asim(x, N)

x = np.linspace(x_min, x_max, 10000)

N = 12
taylor = Ai_taylor(x, N)
pos = Ai_pos_asim(x, N)
neg = Ai_neg_asim(x, N)

AI = [Ai(xi, N) for xi in x]
Ai_real = [airy(xi)[0] for xi in x]


plt.plot(x, Ai_real, label='real')
plt.plot(x, AI, label='approx')
plt.legend()
plt.ylim(-5, 5)
plt.show()
