# 
# 
# Egil Furnes 
# egilsf@uio.no
# Final project in IN1900, fall 2024
# 
#

# Problem 2

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# a)

class SEIR:
    def __init__(self, beta=0.33, r_ia=0.1,
                 r_e2=1.25, lmbda_1=0.33,
                 lmbda_2=0.5, p_a=0.4, mu=0.2):
        if isinstance(beta, (float, int)):
            self.beta = lambda t: beta  # Convert constant beta to a callable function
        elif callable(beta):
            self.beta = beta  # Use beta directly if it’s already callable
        else:
            raise ValueError("beta must be float/int/callable function")

        self.r_ia = r_ia
        self.r_e2 = r_e2
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2
        self.p_a = p_a
        self.mu = mu
        
    def __call__(self, t, u):
        beta = self.beta(t)  # Evaluate beta as a function of time
        r_ia = self.r_ia
        r_e2 = self.r_e2
        lmbda_1 = self.lmbda_1
        lmbda_2 = self.lmbda_2
        p_a = self.p_a
        mu = self.mu

        S, E1, E2, I, Ia, R = u
        N = sum(u)
        dS = -beta * S * I / N - r_ia * beta * S * Ia / N - r_e2 * beta * S * E2 / N
        dE1 = beta * S * I / N + r_ia * beta * S * Ia / N + r_e2 * beta * S * E2 / N - lmbda_1 * E1
        dE2 = lmbda_1 * (1 - p_a) * E1 - lmbda_2 * E2
        dI = lmbda_2 * E2 - mu * I
        dIa = lmbda_1 * p_a * E1 - mu * Ia
        dR = mu * (I + Ia)
        return [dS, dE1, dE2, dI, dIa, dR] 
    
# b) 

def test_SEIR_beta_const():
    model = SEIR(beta=0.33)
    t = 0
    u = [1, 1, 1, 1, 1, 1]

    output = model(t, u)
    beta, r_ia, r_e2 = model.beta(t), model.r_ia, model.r_e2
    lmbda_1, lmbda_2, p_a, mu = model.lmbda_1, model.lmbda_2, model.p_a, model.mu

    S, E1, E2, I, Ia, R = u
    N = sum(u)

    expected = [
        -beta * S * I / N - r_ia * beta * S * Ia / N - r_e2 * beta * S * E2 / N,
        beta * S * I / N + r_ia * beta * S * Ia / N + r_e2 * beta * S * E2 / N - lmbda_1 * E1,
        lmbda_1 * (1 - p_a) * E1 - lmbda_2 * E2,
        lmbda_2 * E2 - mu * I,
        lmbda_1 * p_a * E1 - mu * Ia,
        mu * (I + Ia)
    ]

    tol = 1e-10
    for i, (computed, exp) in enumerate(zip(output, expected)):
        assert abs(computed - exp) < tol

    print(f"test_SEIR_beta_const passed with a toleranse of {tol}")

def test_SEIR_beta_var():
    def betafunction(t):
        return 0.5 if t < 10 else 0.2

    model = SEIR(beta=betafunction)
    t1, t2 = 5, 15
    u = [1, 1, 1, 1, 1, 1]

    output1 = model(t1, u)
    output2 = model(t2, u)

    beta1 = betafunction(t1)
    S, E1, E2, I, Ia, R = u
    N = sum(u)

    expected1 = [
        -beta1 * S * I / N - model.r_ia * beta1 * S * Ia / N - model.r_e2 * beta1 * S * E2 / N,
        beta1 * S * I / N + model.r_ia * beta1 * S * Ia / N + model.r_e2 * beta1 * S * E2 / N - model.lmbda_1 * E1,
        model.lmbda_1 * (1 - model.p_a) * E1 - model.lmbda_2 * E2,
        model.lmbda_2 * E2 - model.mu * I,
        model.lmbda_1 * model.p_a * E1 - model.mu * Ia,
        model.mu * (I + Ia)
    ]

    beta2 = betafunction(t2)
    expected2 = [
        -beta2 * S * I / N - model.r_ia * beta2 * S * Ia / N - model.r_e2 * beta2 * S * E2 / N,
        beta2 * S * I / N + model.r_ia * beta2 * S * Ia / N + model.r_e2 * beta2 * S * E2 / N - model.lmbda_1 * E1,
        model.lmbda_1 * (1 - model.p_a) * E1 - model.lmbda_2 * E2,
        model.lmbda_2 * E2 - model.mu * I,
        model.lmbda_1 * model.p_a * E1 - model.mu * Ia,
        model.mu * (I + Ia)
    ]

    tol = 1e-10
    for i, (computed, exp) in enumerate(zip(output1, expected1)):
        assert abs(computed - exp) < tol
    for i, (computed, exp) in enumerate(zip(output2, expected2)):
        assert abs(computed - exp) < tol

    print(f"test_SEIR_beta_var passed with a tolerance of {tol}!")

# run tests
test_SEIR_beta_const()
test_SEIR_beta_var()

# c)

def solve_SEIR(T,dt,S_0,E2_0,beta):
    model = SEIR(beta=beta)
    initial = [S_0, 0, E2_0, 0, 0, 0]
    timeval = np.arange(0, T+dt, dt)
    solution = solve_ivp(model, (0,T) , initial, t_eval = timeval, method='RK45')
    t = solution.t
    u = solution.y.T
    return t, u

# d)

def plot_SEIR(t, u, components=['S','I','Ia','R']):

    component_map = {
        'S': 0,
        'E1': 1,
        'E2': 2,
        'I': 3,
        'Ia': 4,
        'R': 5
    }
    
    plt.figure(figsize=(10, 6))

    for component in components:
        if component in component_map:
            plt.plot(t, u[:, component_map[component]], label=f"{component}(t)")
        else:
            raise ValueError(f"non-valid component {component}")

    plt.xlabel('time')
    plt.ylabel('population')
    plt.legend()
    plt.grid(True)
    plt.title('SEIR model')
    plt.show()

# e)

if __name__ == "__main__":
    t, u = solve_SEIR(T=300, dt=1.0, S_0=5.5e6, E2_0=100, beta=0.4)
    plot_SEIR(t, u)
    smittetopp = max(u[:,3])
    print(f'max antall smittede {smittetopp:.2f}')

