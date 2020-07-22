#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

ind = np.arange(3)
width = 0.5
p1 = plt.bar(ind, fruit[0], width, color='red')
p2 = plt.bar(ind, fruit[1], width, bottom=fruit[0], color='yellow')
p3 = plt.bar(ind, fruit[2], width, bottom=fruit[0] + fruit[1], color='#ff8000')
p4 = plt.bar(ind, fruit[3], width, bottom=fruit[0] + fruit[1] + fruit[2], color='#ffe5b4')


plt.title('Number of Fruit per Person')
plt.xticks(ind, ('Farrah', 'Fred', 'Felicia'))
plt.ylabel('Quantity of Fruit')
plt.yticks(np.arange(0, 90, 10))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('apples', 'bananas', 'oranges', 'peaches'))

plt.show()
