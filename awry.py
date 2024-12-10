
import numpy as np
import os

os.system('cls' if os.name == 'nt' else 'clear')

x = np.linspace(0, 10, 101)
print(x)

a = np.zeros(x.shape, x.dtype)
print(a)

a = x.copy()
print(a)

a = np.zeros_like(x)
print(a)

a = np.linspace(1, 8, 8)
a < 0
print(a<0)
print(a[a<0])

from numpy import *

A = zeros((3,4))
print(A)

A[0,0] = -1
A[1,0] = 1
A[2,0] =   10
A[0,1] = -5
A[2,3] = -100
A[2][3] = -100

print(A)

