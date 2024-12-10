
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

# t = np.linspace(0, 8, 201)
# plt.plot(x, np.exp(-x)*np.sin(2*np.pi*x), x, np.exp(-2*x)*np.sin(4*np.pi*x))
# plt.show()

from numpy import *
import matplotlib.pyplot as plt

# formula = input('write a mathematical expression using x: ')
# xmin = float(input('lower bound for x: '))
# xmax = float(input('upper bound for x: '))

# x = linspace(xmin, xmax, 101)
# y = eval(formula)
# print(y)

# plt.plot(x, y)
# plt.show()

print(f'\n')

def H(x):
    if x < 0:
        return 0
    elif x >= 0: 
        return 1

print(H(0))
print(H(1))
print(H(2))

x = linspace(-10, 10, 5)
# y = H(x)
# plt.plot(x, y)
# plt.show()

n = 5
x = np.linspace(-5, 5, n+1)
y = np.zeros(n+1)

for i in range(len(x)):
    y[i] = H(x[i])

# plt.plot(x,y)
# plt.show()

def Hl(x):
    r = np.zeros(len(x))
    for i in range(len(x)):
        r[i] = H(x[i])
    return r

n = 5
x = np.linspace(-5, 5, n+1)
y = Hl(x)

# plt.plot(x,y)
# plt.show()

Hv = np.vectorize(H)
def Hv(x): return np.where(x<0, 0.0, 1.0)

# def fvec(x): 
#     x1 = <exp1>
#     x2 = <exp2>
#     r = np.where(condition, x1, x2)
#     return r

def f(x,m,s):
    return (1.0/(np.sqrt(2*np.pi*pi)*s))*np.exp(-0.5*((x-m)/s)**2)

m = 0; s_start = 2; s_stop = 0.2
s_values = np.linspace(s_start, s_stop, 30)

x = np.linspace(m-3*s_start, m+3*s_start, 1000)
max_f = f(m, m, s_stop)

y = f(x,m,s_stop)
lines = plt.plot(x,y)

plt.axis([x[0], x[-1], -0.1, max_f])
plt.xlabel('x')
plt.ylabel('y')

# for s in s_values:
#     y = f(x, m, s)
#     lines[0].set_ydata(y)
#     plt.draw()
#     plt.pause(0.1)


from matplotlib.animation import FuncAnimation
def f(x,m,s):
    return (1.0/(np.sqrt(2*np.pi*pi)*s))*np.exp(-0.5*((x-m)/s)**2)

m = 0; s_start = 2; s_stop = 0.2
s_values = np.linspace(s_start, s_stop, 30)
x = np.linspace(-3*s_start, 3*s_start, 1000)
max_f = f(m, m, s_stop)

plt.axis([x[0], x[-1], 0, max_f])
plt.xlabel('x'); plt.ylabel('y')

y = f(x, m, s_start)
lines = plt.plot(x, y)

def next_frame(s):
    y = f(x, m, s)
    lines[0].set_ydata(y)
    return lines

# ani = FuncAnimation(plt.gcf(), next_frame, frames=s_values, interval=1000)
# ani.save('movie.gif', fps=60)
# # plt.show()


