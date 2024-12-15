

import numpy as np

class ForwardEuler: 
    def __init__(self, f):
        self.f = lambda t, u: np.asarray(f(t, u), float)

    def set_initial_condition(self, u0):
        if np.isscalar(u0):
            self.neq = 1
            u0 = float(u0)
        else:
            self.neq = u0.size
            u0 = np.asarray(u0)
        self.u0 = u0

    def solve(self, t_span, N):
        t0, T = t_span
        self.dt = (T-t0)/N
        self.t = np.zeros(N+1)
        if self.neq == 1:
            self.u = np.zeros(N+1)
        else:
            self.u = np.zeros((N+1, self.neq))

        msg = 'please set initial condition'
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
        return u[n] + dt*(f(t[n], u[n]))
    
from math import sin

class Pendulum:
    def __init__(self, L, g=9.81):
        self.L = L
        self.g = g
    
    def __call__(self, t, u):
        theta, omega = u
        dtheta = omega
        domega = -self.g / self.L * sin(theta)
        return [dtheta, domega]

# print(Pendulum(L=10))

# from matplotlib import pyplot as plt
# problem = Pendulum(L=1)
# solver = ForwardEuler(problem)
# # solver.set_initial_condition([np.pi / 4, 0])
# solver.set_initial_condition([np.pi / 4, 0])
# T = 10
# N = 1000
# t, u = solver.solve(t_span=(0,T), N=N)

# plt.plot(t, u[:,0], lable=r'$\theta$')
# plt.plot(t, u[:,1], lable=r'$\omega$')
# plt.xlabel('t')
# plt.ylabel(r'angle ($\theta$) and regular velocity ($\omega$)')
# plt.legend()
# plt.show()





