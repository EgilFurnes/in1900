
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import os

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')
clear_terminal()
print(f'\n')
print("clean terminus")

# numpy arrays

print("hello")

def f(x):
    return x**2

n = 5; dx = 1.0/(n-1)
xlist = [i*dx for i in range(n)]; ylist = [f(x) for x in xlist]

x = np.array(xlist)
y = np.array(ylist)

print(x, y)

import numpy as np
def f(x): return x**2
n = 5
x = np.linspace(0,1,n)
y = np.zeros(n)
for i in range(n):
    y[i] = f(x[i])

print(x, y)

print(np.zeros(10))

import numpy as np
from math import cos
x = np.linspace(0,1,11)
# for i in range(len(x)): y[i] = cos(x[i])
print(y)
y = np.cos(x)
print(y)

from numpy import sin, exp, linspace

def g(x):
    return x**2 + 2*x - 4

def f(x):
    return sin(x)*exp(-2*x)

x = 1.2
y = f(x)

x = linspace(0, 3, 101)
y = f(x)
z = g(x)

import math, numpy
x = numpy.linspace(0, 1, 6)

print(x)
print(math.cos(x[0]))

# print(math.cos(x))
# print(numpy.cos(x))

import numpy as np

n = 100
x = np.linspace(0, 4, n+1)
y = np.exp(-x)*np.sin(2*np.pi*x)

print(n, x, y)

from matplotlib import pyplot as plt
import numpy as np

n = 100
x = np.linspace(0, 4, n+1)
y = np.exp(-x)*np.sin(2*np.pi*x)

# plt.plot(x, y)
# plt.show()

def f(x): return np.exp(-x)*np.sin(2*np.pi*x)
n = 100; x = linspace(0, 4, n+1); y = f(x)

# plt.plot(x, y, label='exp(-x)*sin(2$\pi$x)')
# plt.xlabel('x'); plt.ylabel('y')
# plt.legend()
# plt.axis([0, 4, -0.5, 0.8])
# plt.title('my first matplotlib demo')
# plt.savefig('fig.pdf'); plt.savefig('fig.png'); 
# plt.show()


def f1(x): return np.exp(-x)*np.sin(2*np.pi*x)
def f2(x): return np.exp(-2*x)*np.sin(4*np.pi*x)

x = np.linspace(0,8,401)
y1 = f1(x)
y2 = f2(x)

# plt.plot(x, y1, 'r--', label='exp(-x)*sin(2$\pi$x)')
# plt.plot(x, y2, 'g:', label='exp(-2x)*sin(4$\pi$x)')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

