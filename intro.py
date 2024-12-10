
n = 10
amount = 100
print(f"After {n} years, it grew by {amount} EUR.")
print(f"{2+8}")

t = 1.234567
p = 180942380234.42234234
print(f"Default output gives t = {t}.")
print(f"We can set the precision t = {t:.2}.")
print(f"Or control number of decimals t = {t:.2f}.")
print(f"we may set the space used to t = {t:8.2f}")
print(f"most compact form t = {t:g}")
print(f"most compact form t = {p:g}")

import math
r = math.sqrt(2)
from math import sqrt
r = sqrt(2)
from math import *
r = sqrt(2)
print(r)

from math import sqrt, pi, exp
m = 0; s = 2; x = 1.0
print(m,s)

f = 1/(sqrt(2*pi*s))*exp(-0.5*((x-m)/(s)**2))
print(f)

import math
print(dir(math))
print(remainder(10,4))

for i in range(10):
    if remainder(i, 2) == 0:
        print("even")
    else:
        print("odd")

v1 = 1/49.0*49
v2 = 1/51.0*51
print(f"{v1:.16f} {v2:.16f}")

N = 14
S = 0
for i in range(1, N+1):
    S += i**2
print(S)


