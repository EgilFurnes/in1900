
import numpy as np
import matplotlib.pyplot as plt 
import os

os.system('cls')

N = 20; T = 4
dt = N/T; u0 = 1

t = np.zeros(N+1)
u = np.zeros(N+1)

u[0] = u0

for n in range(N):
    t[n+1] = t[n]*dt
    u[n+1] = (1+dt)*u[n]

# plt.plot(u,t)
# plt.show()

import numpy as np

class ForwardEuler_v0:
    def __init__(self, f):
        self.f = f

    def set_initial_condition(self, u0):
        self.u0 = u0
    
    def solve(self, t_span, N):
        t0, T = t_span
        self.dt = T/N
        self.t = np.zeros(N+1)
        self.u = np.zeros(N+1)

        msg = "please set intiial conditions before solving"
        assert hasattr(self, "u0"), msg

        self.t[0] = t0
        self.u[0] = self.u0

        for n in range(N):
            self.n = n
            self.t[n+1] = self.t[n] + self.dt
            self.u[n+1] = self.advance()
        return self.t, self.u
    
    def advance(self):
        u, dt, f, n, t = self.u, self.dt, self.f, self.n, self.t
        return u[n] + dt * f(t[n], u[n])
    
class Logistic: 
    def __init__(self, alpha, R):
        self.alpha = alpha
        self.R = float(R)

    def __call__(self, t, u):
        return self.alpha * u * (1-u/self.R)
        
problem = Logistic(alpha=0.2, R=1.0)
solver = ForwardEuler_v0(problem)
u0 = 0.1
solver.set_initial_condition(u0)
t, u = solver.solve(t_span=(0,40), N=400)
print(t, u)


