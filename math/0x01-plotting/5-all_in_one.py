#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, ax = plt.subplots()
ax.set_title("All in One", va='bottom')
plt.subplots_adjust(hspace=1, wspace=0.37)
plt.suptitle('All in One')

plt.subplot(321)
plt.plot(y0, 'r')
plt.axis([0, 10, -50, 1050])

plt.subplot(322)
plt.scatter('a', 'b', c='magenta', data = {'a': x1, 'b': y1}, s = 5)
plt.title("Men's Height vs Weight", size="x-small")
plt.xlabel('Height (in)', size="x-small")
plt.ylabel('Weight (lbs)', size="x-small")

plt.subplot(323)
plt.plot(x2, y2)
plt.yscale('log')
plt.xlim(0, 28500)
plt.title('Exponential Decay of C-14', size="x-small")
plt.xlabel('Time (years)', size="x-small")
plt.ylabel('Fraction Remaining', size="x-small")


plt.subplot(324)
plt.plot(x3, y31, 'r--', label='C-14')
plt.plot(x3, y32, 'g', label='Ra-226')
plt.axis([0, 20000, 0.0, 1.0])
plt.yticks(np.arange(0.0, 1.5, step=0.5))
plt.title('Exponential Decay of Radiactive Elements', size="x-small")
plt.xlabel('Time (years)', size="x-small")
plt.ylabel('Fraction Remaining', size="x-small")
plt.legend(framealpha=1, frameon=True, fontsize="x-small")


plt.subplot(313)
plt.hist(student_grades, edgecolor='black', bins=range(0, 110, 10))
plt.axis([0, 100, 0, 30])
plt.xticks(np.arange(0, 110, step=10))
plt.title('Project A', size="x-small")
plt.xlabel('Grades', size="x-small")
plt.ylabel('Number of Students', size="x-small")

plt.show()
