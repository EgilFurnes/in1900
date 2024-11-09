
import numpy as np
from math import *

# langtangen: 2.4 odd
def odd(n):
    i = 1
    while i < n:
        print(i)
        i+=2
odd(10)

# langtangen: 2.7 coor
print(f"\n")
def coor(n, a, b):
    space=[]
    h = (b-a)/n
    for i in range(n+1):
        xi = a+i*h
        space.append({xi})
    print(f"{space}")

coor(10, 4.3, pi)

# langtangen: 2.8 ball_table1

def ball_table(v0):
    g = 9.81 
    t = 0
    dt = 0.1
    while t <= 2*v0/g:
        yt = v0*t-0.5*g*t**2
        print(f"{t:10e} | {yt:10e}")
        t += dt

print(f"\n")
ball_table(5)

def forball(v0, n):
    g = 9.81
    tmax = 2*v0 / g
    h = tmax/n
    for i in range(n+1):
        t = i*h
        y = v0*t-0.5*g*t**2
        print(f"{t:4.3f} | {y:4.3f}")

print(f"\n")
forball(5, 10)

def whileball(v0, n):
    g = 9.81
    tmax = 2*v0/g
    h = tmax/n
    i = 1
    while i < n+1:
        t = i*h
        y = v0*t-0.5*g*t**2
        i+=1
        print(f"{t:4.3f} | {y:4.3f}")

print(f"\n")
whileball(5, 10)

# langtangen: 2.14 inverse_sine

print(f"\n")
print(asinh(10))

# langtangen: 2.15 index_nested_list

q = [['a','b','c'], ['d','e','f'], ['g','h']]
print(q[0][0], q[1], q[-1][-1], q[1][0], q[-1][-2])

for i in q:
    for j in range(len(i)):
        print(i[j])
print(type(i), type(j))

# langtangen: 3.20 hw_func
print(f"\n")

def hw1():
    return "Hello World!"
print(hw1())

def hw2():
    print("Hello World!")
hw2()

def hw3(a, b):
    return f"{a}{b}"
print(hw3("Hello, ", "World!"))
print(hw3('Python ', 'function'))

# langtangen: 3.23 egg_func

print(f"\n")

def gauss(x, n, m=0, s=1):
    interval = np.linspace(m-5*s, m+5*s, n)
    for x in interval:
        fx = 1/(sqrt(2*pi)*s)*exp(-0.5*((x-m)/(s))**2)
        print(f"{x:5.5f} | {fx:5.5f}")

gauss(x=10, n=10)

import matplotlib.pyplot as plt

def gaussplot(n, m=0, s=1):
    x = np.linspace(m-5*s, m+5*s, n)
    fx = 1/(np.sqrt(2*pi)*s)*np.exp(-0.5*((x-m)/(s))**2)
    plt.plot(x, fx)
    plt.show()

# gaussplot(n=100)
print(f"\n")

# langtangen: 3.28 maxmin_list

a = [1,52,3,14,5,36,7,8,9,10]
max_elem = a[0]
for i in a[1:]:
    if max_elem < i:
        max_elem = i

min_elem = a[0]
for i in a[1:]:
    if min_elem > i:
        min_elem = i

print(f"for list {a} the max is {max_elem} and min is {min_elem}")

# oppgaveheftet: 3.4 sum_for



# oppgaveheftet: 3.5 sum_while
# oppgaveheftet: 3.7 population_table
# oppgaveheftet: 3.8 population_table2
# oppgaveheftet: 3.11 alkane

# oppgaveheftet: 4.2 interest_rate_loop
# oppgaveheftet: 4.3 factorial
# langtangen: 2.6 energy_levels 
# langtangen: 3.4 fc2
# kjemi: 3.1 nernst_function
# kjemi: 3.3 pH-titration



