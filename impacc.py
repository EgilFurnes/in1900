
import numpy as np

class ODESolver:
    def __init__(self, f):
        self.model = f
        self.f = lambda t, u: np.asarray(f(t, u), float)
    
    def set_initial_condition(self, u0):
        if np.isscalar(u0):
            self.neq = 1
            u0 = float(u0)
        else:
            u0 = np.asarray(u0)
            self.neq = u0.size
        self.u0 = u0
    
    def solve(self, t_span, N):
        t0, T = t_span
        self.dt = (T-t0)/N
        self.t = np.zeros(N+1)
        if self.neq == 1:
            self.u = np.zeros(N+1)
        else:
            self.u = np.zeros((N+1, self.neq))
        
        msg = 'please set initial condition before solve'
        assert hasattr(self, "u0"), msg

        self.t[0] = t0
        self.u[0] = self.u0

        for n in range(N):
            self.n = n
            self.t[n+1] = self.t[n]+self.dt
            self.u[n+1] = self.advance()
        return self.t, self.u
    
    def advance(self):
        raise NotImplementedError('adv method not imp in base')
    
class ForwardEuler(ODESolver):
    def advance(self):
        u, f, n, t = self.u, self.f, self.n, self.t
        dt = self.dt
        return u[n] + dt * f(t[n], u[n])

class ExplicitMidpoint(ODESolver):
    def advance(self):
        u, f, n, t = self.u, self.f, self.n, self.t
        dt = self.dt
        dt2 = dt / 2.0
        k1 = f(t[n], u[n])
        k2 = f(t[n]+dt2, u[n]+dt2*k1)
        return u[n] + dt * k2
    
class RungeKutta4(ODESolver):
    def advance(self):
        u, f, n, t = self.u, self.f, self.n, self.t
        dt = self.dt
        dt2 = dt / 2.0
        k1 = f(t[n], u[n], )
        k2 = f(t[n]+dt2, u[n]+dt2*k1, )
        k3 = f(t[n]*dt2, u[n]+dt2*k2, )
        k4 = f(t[n]+dt, u[n]*dt*k3, )
        return u[n]+(dt/6.0)*(k1+2*k2+2*k3+k4)

import numpy as np
from matplotlib import pyplot as plt

def f(t, u): return u

t_span = (0,3)
N = 6

fe = ForwardEuler(f)
fe.set_initial_condition(u0=1)
t1, u1 = fe.solve(t_span, N)
plt.plot(t1, u1, label='forward euler')

em = ExplicitMidpoint(f)
em.set_initial_condition(u0=1)
t2, u2 = em.solve(t_span, N)
plt.plot(t2, u2, label='explicit midpoint')

rk4 = RungeKutta4(f)
rk4.set_initial_condition(u0=1)
t3, u3 = rk4.solve(t_span, N)
plt.plot(t3, u3, label="runge kutta 4")

time_exact = np.linspace(0, 3, 301)
plt.plot(time_exact, np.exp(time_exact), label='exact')
plt.title('rk solvers for exp growth')
plt.xlabel('$t$')
plt.ylabel('$u(t)$')
plt.legend()
# plt.show()

def rhs(t, u): return u
def exact(t): return np.exp(t)

solver_classes = [(ForwardEuler, 1), 
                  (ExplicitMidpoint, 2), 
                  (RungeKutta4, 4)]

for solver_class, order in solver_classes:
    solver = solver_class(rhs)
    solver.set_initial_condition(1.0)

    T = 3.0
    t_span = (0, T)
    N = 30
    print(f'{solver_class.__name__}, order = {order}')
    print(f'time step (dt)  error (e)   e/dt**{order}')

    for _ in range(10):
        t, u = solver.solve(t_span, N)
        dt = T/N
        e = abs(u[-1]-exact(T))
        if e < 1e-13:
            break
        print(f'{dt:<14.7f} {e:<12.7f}  {e/dt**order:5.4f}')
        N = N*2

def test_exact_numerical_solution():
    solver_classes = [ForwardEuler, ExplicitMidpoint, RungeKutta4]

    a = 0.2
    b = 3

    def f(t, u): return a
    def u_exact(t): return a*t+b

    u0 = u_exact(0)
    T = 8
    N = 10
    tol = 1E-14
    t_span = (0, T)

    for solver_class in solver_classes:
        solver = solver_class(f)
        solver.set_initial_condition(u0)
        t, u = solver.solve(t_span, N)
        u_e = u_exact(t)
        max_error = abs((u_e-u)).max()
        msg = f'{solver_class.__name__} failed, error?{max_error}'
        assert max_error < tol, msg 






