#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

fig, ax = plt.subplots()
plt.scatter('a', 'b', c='magenta', data = {'a': x, 'b': y}, s = 5)
ax.set_title("Men's Height vs Weight")
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.show()
