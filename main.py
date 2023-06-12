import numpy as np
import random
import time
from scipy.stats import norm

import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.animation import FuncAnimation, PillowWriter


# пункт номер 1

numpy_array1 = np.array(np.random.randint(1, 10, 10**6 + 1), int)
numpy_array2 = np.array(np.random.randint(1, 10, 10**6 + 1), int) # создаём numpy масивы и наполняем

standard_array1 = [random.randint(1, 10) for i in range(10**6 + 1)]
standard_array2 = [random.randint(1, 10) for j in range(10**6 + 1)] # создаём простые масивы и наполняем


#перемнодает два обычных масива и замеряет время.
array_of_products = []
for i in range(10**6 + 1):
    array_of_products.append(standard_array1[i] * standard_array2[i])
print(time.perf_counter()) #вывод времени

#перемнодает два масива вида Numpyо и замеряет время.
result_numpy = np.multiply(numpy_array1, numpy_array2)
print(time.perf_counter()) #вывод времени.


#пункт номер 2 вариант 8
columns = ['Chloramines']
df = pd.read_csv('data2.csv', usecols=columns)
_, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_xlabel('Chloramines')
ax1.set_ylabel('count')
ax2.set_xlabel('Chloramines')
ax2.set_ylabel('count')
ax1.set_title('default histogram')
ax2.set_title('normalized histogram')
ax1.hist(df.Chloramines)
ax2.hist(df.Chloramines, density=True)
mean, std = norm.fit(df.Chloramines)
x = np.linspace(df.Chloramines.min(), df.Chloramines.max())
ax2.plot(x, norm.pdf(x, mean, std))
plt.show()



#Задание 3 (формула: x∈(-3п;3п); y=cos(x); z=x/sin(x))

x = np.linspace(-3 * np.pi, 3 * np.pi, 50)
y = np.cos(x)
z = x / np.sin(x)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='parametric curve')
plt.show()

# дополнительное задание

def animated_plot():
    def animate(i):
        x = np.linspace(-2 * np.pi, 2 * np.pi, 300)
        y = np.sin(x * i)
        ax.clear()
        return ax.plot(x, y)
    fig, ax = plt.subplots()
    animation = FuncAnimation(fig, animate, frames= np.linspace(-2 * np.pi, 2 * np.pi, 30), interval = 10, repeat=False)
    animation.save('sin_animation.gif', writer=PillowWriter(fps=30))

animated_plot()