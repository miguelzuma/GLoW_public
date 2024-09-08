import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 4.5*np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots()
fig.suptitle('sample plot')
ax.plot(x, y)

plt.show()
