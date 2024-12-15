
import sys
from numpy import zeros

N = 100000
x = zeros(N+1, int)
x[0] = 1
x[1] = 1
# for n in range(2, N+1):
#     x[n] = x[n-1] + x[n-2]
#     print(n, x[n])

from matplotlib import pyplot as plt

x0 = 100
rho = 5
R = 500
N = 200

index_set = range(N+1)
x = zeros(len(index_set))
x[0] = x0
for n in index_set[1:]:
    x[n] = x[n-1] + (rho/100)*x[n-1]*(1-x[n-1]/R)

# plt.plot(index_set, x)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

def Newton(f, dfdx, x, epsilon = 1.0E-7, max_n = 100):
    n = 0
    while abs(f(x)) > epsilon and n < max_n:
        x = x-f(x)/dfdx(x)
        n += 1
    return x, n, f(x)

import numpy as np
 
F = 1e7
p = 5
I = 3
q = 75
N = 40
index_set = range(N+1)
x = np.zeros(len(index_set))
c = np.zeros_like(x)

x[0] = F
c[0] = q*p*F*1e-4

for n in index_set[1:]:
    x[n] = x[n-1]+(p/100.0)*x[n-1]-c[n-1]
    c[n] = c[n-1]+(I/100.0)*c[n-1]

# plt.plot(index_set, x, 'ro', label='fortune')
# plt.plot(index_set, c, 'go', label='yearly consume')
# plt.xlabel('years')
# plt.ylabel('amount')
# plt.legend()
# # plt.show()

import numpy as np
import matplotlib.pyplot as plt

x0 = 100
y0 = 8
a = 0.0015
b = 0.003
c = 0.006
d = 0.5
N = 10000
index_set = range(N+1)
x = np.zeros(len(index_set))
y = np.zeros_like(x)

x[0] = x0
y[0] = y0

for n in index_set[1:]:
    x[n] = x[n-1]+a*x[n-1]-b*x[n-1]*y[n-1]
    y[n] = y[n-1]+d*b*x[n-1]*y[n-1]-c*y[n-1]

# plt.plot(index_set, x, label='prey')
# plt.plot(index_set, y, label='predator')
# plt.xlabel('time')
# plt.ylabel('pop')
# plt.legend()
# plt.show()

import numpy as np
x = 0.5
N = 5
index_set = range(N+1)
a = np.zeros(len(index_set))
e = np.zeros(len(index_set))
a[0] = 1

print(f'exact: exp({x}) = {np.exp(x)}')
for n in index_set[1:]:
    e[n] = e[n-1] + a[n-1]
    a[n] = x/n*a[n-1]
    print(f'n={n:4.5f}, appr = {e[n]:4.5f}, e = {np.abs(e[n]-np.exp(x)):4.5f}')


