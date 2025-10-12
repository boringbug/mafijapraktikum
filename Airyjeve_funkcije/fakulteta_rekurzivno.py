import matplotlib.pyplot as plt
import numpy as np

def taylor_exp(x, n):
    sum = 1.0
    prod = 1.0
    for i in range(1, n):
        prod *= 1/i
        sum += np.pow(x, i)*prod
    return sum


x = np.linspace(-3, 3, 100000)
y = np.exp(x)
y_taylor = taylor_exp(x, 1000)

for n in range(2, 5):
    y_n = taylor_exp(x, n)
    #plt.plot(x, y_n, label='n = ' + str(n))

#plt.plot(x, y, label='original')
#plt.plot(x, y_taylor, label='taylor')
plt.plot(x, y - y_taylor)
plt.grid(True)
plt.ylim(-3, 3)
plt.legend()
plt.show()


