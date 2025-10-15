import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
from scipy.special import airy
from definicije import Ai_taylor, Ai_pos_asim, Ai_neg_asim


N = 5

x = np.linspace(-20, 8, 10000)
x_pos = np.linspace(8, 1e5, 10000)
x_neg = np.linspace(-1e2, -20, 10000)

x_line = np.linspace(-50, 10, 10000)
line = [1e-10 for i in x_line]

# Calculating and ploting the uncuritanty
diff_taylor = abs(Ai_taylor(x, 10) - airy(x)[0])
diff_pos_asim = abs(Ai_pos_asim(x_pos, 10) - airy(x_pos)[0])
diff_neg_asim = abs(Ai_neg_asim(x_neg, 10) - airy(x_neg)[0])

plt.title("Abslutna napaka aproksimacij")
plt.plot(x, diff_taylor, label='taylor abs diff')
plt.plot(x_pos, diff_pos_asim, label='pos asim abs diff')
plt.plot(x_neg, diff_neg_asim, label='neg asim abs diff')
plt.plot(x_line, line, color='red')
plt.grid()
plt.ylim(0, 1e-9)
plt.legend()
plt.show()

# Semi log graph
plt.title("Abslutna napaka aproksimacij v semi-log")
plt.semilogy(x, diff_taylor, label='taylor abs diff')
plt.semilogy(x_pos, diff_pos_asim, label='pos asim abs diff')
plt.plot(x_line, line, color='red')
plt.semilogy(x_neg, diff_neg_asim, label='pos asim abs diff')
plt.grid()
plt.legend()
plt.show()

# Log-Log graph
plt.title("Abslutna napaka aproksimacij v log-log")
plt.loglog(x, diff_taylor, label='taylor abs diff')
plt.loglog(x_pos, diff_pos_asim, label='pos asim abs diff')
plt.loglog(-x_neg, diff_neg_asim, label='neg asim abs diff')
plt.plot(x_line, line, color='red')
plt.grid()
plt.legend()
plt.show()
