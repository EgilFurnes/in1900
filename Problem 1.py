# 
# 
# Egil Furnes 
# egilsf@uio.no
# Final project in IN1900, fall 2024
# 
#

# Problem 1 

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class SEIR0:
    def __init__(self, beta=0.33, r_ia=0.1,
                 r_e2=1.25, lmbda_1=0.33,
                 lmbda_2=0.5, p_a=0.4, mu=0.2):

        self.beta = beta
        self.r_ia = r_ia
        self.r_e2 = r_e2
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2
        self.p_a = p_a
        self.mu = mu

    def __call__(self, t, u):
        beta = self.beta
        r_ia = self.r_ia
        r_e2 = self.r_e2
        lmbda_1 = self.lmbda_1
        lmbda_2 = self.lmbda_2
        p_a = self.p_a
        mu = self.mu

        S, E1, E2, I, Ia, R = u
        N = sum(u)
        dS = -beta * S * I / N - r_ia * beta * S * Ia / N \
            - r_e2 * beta * S * E2 / N
        dE1 = beta * S * I / N + r_ia * beta * S * Ia / N \
            + r_e2 * beta * S * E2 / N - lmbda_1 * E1
        dE2 = lmbda_1 * (1 - p_a) * E1 - lmbda_2 * E2
        dI = lmbda_2 * E2 - mu * I
        dIa = lmbda_1 * p_a * E1 - mu * Ia
        dR = mu * (I + Ia)
        return [dS, dE1, dE2, dI, dIa, dR]

# a)

def test_SEIR0():
    model = SEIR0()
    t = 0
    u = [1,1,1,1,1,1]
    output = model(t,u)
    beta, r_ia, r_e2 = model.beta, model.r_ia, model.r_e2
    lmbda_1, lmbda_2, p_a, mu = model.lmbda_1, model.lmbda_2, model.p_a, model.mu
    S, E1, E2, I, Ia, R = u
    N = sum(u)

    expected = [
        # dS
        -beta * S * I / N - r_ia * beta * S * Ia / N - r_e2 * beta * S * E2 / N,  
        # dE1
        beta * S * I / N + r_ia * beta * S * Ia / N + r_e2 * beta * S * E2 / N - lmbda_1 * E1,  
        # dE2
        lmbda_1 * (1 - p_a) * E1 - lmbda_2 * E2,  
        # dI
        lmbda_2 * E2 - mu * I,  
        # dIa
        lmbda_1 * p_a * E1 - mu * Ia,  
        # dR
        mu * (I + Ia)  
    ]

    tol = 1e-10
    for i, (computed, exp) in enumerate(zip(output, expected)):
        assert abs(computed-exp)<tol 
    print(f"testene passerte med en margin på {tol}")

test_SEIR0()

# b)

def solve_SEIR(T,dt,S_0,E2_0,beta):
    model = SEIR0(beta=beta)
    initial = [S_0, 0, E2_0, 0, 0, 0]
    timeval = np.arange(0, T+dt, dt)
    solution = solve_ivp(model, (0,T) , initial, t_eval = timeval, method='RK45')
    t = solution.t
    u = solution.y.T
    return t, u

# c)

def plot_SEIR(t,u):
    plt.figure(figsize=(10,6))
    plt.plot(t,u[:,0], label='S(t)')
    plt.plot(t,u[:,3], label='I(t)')
    plt.plot(t,u[:,4], label='Ia(t)')
    plt.plot(t,u[:,5], label='R(t)')
    plt.xlabel('time in days')
    plt.ylabel('population')
    plt.legend()
    plt.grid(True)
    plt.title('SEIR model')
    plt.show()
    
T, dt, S_0, E2_0, beta = 150, 1.0, 5.5e6, 100, 0.4
t, u = solve_SEIR(T, dt, S_0, E2_0, beta)
plot_SEIR(t, u)
