#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, ax = plt.subplots()
plt.hist(student_grades, edgecolor='black', bins=range(0, 110, 10))
plt.axis([0, 100, 0, 30])
plt.xticks(np.arange(0, 110, step=10))
ax.set_title("Project A", va='bottom')
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.show()
