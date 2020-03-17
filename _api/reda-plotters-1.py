import matplotlib.pyplot as plt
import numpy as np
from reda.plotters import matplot
a = np.arange(4)
b = np.arange(3) + 3
def sum(a, b):
   return a + b
x, y = np.meshgrid(a, b)
c = sum(x, y)
fig, (ax1, ax2) = plt.subplots(1, 2)
im = ax1.pcolormesh(x, y, c)
_ = plt.colorbar(im, ax=ax1)
_ = ax1.set_title("plt.pcolormesh")
_, _ = matplot(x, y, c, ax=ax2)
_ = ax2.set_title("reda.plotters.matplot")
fig.show()
