import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
from scipy.special import airy

x = np.linspace(-20, 2, 1000)
y1, _, y2, _ = airy(x)

plt.plot(x, y1)
plt.plot(x, y2)

plt.grid(True)
plt.show()
