import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
from scipy.special import airy
from aproksimacija_vrsta import Ai, Bi

x = np.linspace(-5, 5, 1000)
Ai_real, _, Bi_real, _ = airy(x)

plt.plot(x, Ai_real)
plt.plot(x, Bi_real)

plt.ylim(-3, 3)

plt.grid(True)
plt.show()
