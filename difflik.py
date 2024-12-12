
# 1.1-1.4, 2.1-2.3 og 5

import numpy as np
from matplotlib import pyplot as plt

N = 20; T = 4; dt = T/N; u0 = 1
t = np.zeros(N+1)
u = np.zeros(N+1)

u[0] = u0
for n in range(N):
    t[n+1] = t[n] + dt
    u[n+1] = (1+dt)*u[n]

# plt.plot(t, u)
# plt.show()

# remember to set u[0] to u0, but no need to set t[0] to t0 somehow (...)

for n in range(1, N+1):
    t[n] = t[n-1]+dt
    u[n] = (1+dt)*u[n-1]

import numpy as np

def forward_euler(f, u0, T, N):
    t = np.zeros(N+1)
    u = np.zeros(N+1)
    u[0] = u0
    dt = T/N

    for n in range(N):
        t[n+1] = t[n] + dt
        u[n+1] = u[n] + dt*f(t[n], u[n])

    return t, u

def f(t, u): return u

u0 = 1; T = 4; N = 30; t, u = forward_euler(f, u0, T, N)

print(t, u)

# plt.plot(t, u)
# plt.show()




