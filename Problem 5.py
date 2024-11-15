# 
# 
# Egil Furnes 
# egilsf@uio.no
# Final project in IN1900, fall 2024
# 
#

# Problem 5

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from lockdown import Beta

class SEIRimport:
    def __init__(self, beta, sigma):
        self.beta = beta  # Instance of Beta class
        self.sigma = sigma  # Import of infections per day
        self.N = 5.5e6  # Total population (can be parameterized if needed)

    def __call__(self, t, u):
        S, E1, E2, I, R, D = u
        pa, mu, gamma, rho = 0.8, 0.01, 0.1, 0.05
        lambda1, lambda2 = 1 / 5.2, 1 / 2.3
        beta_t = self.beta(t)

        dS = -beta_t * S * I / self.N
        dE1 = beta_t * S * I / self.N - lambda1 * E1
        dE2 = lambda1 * (1 - pa) * E1 - lambda2 * E2 + self.sigma
        dI = lambda2 * E2 - (gamma + mu) * I
        dR = gamma * I
        dD = mu * I

        return [dS, dE1, dE2, dI, dR, dD]

def solve_SEIRimport(T, dt, S_0, E2_0, beta, sigma):
    seir_model = SEIRimport(beta, sigma)

    # Initial conditions: S, E1, E2, I, R, D
    u0 = [S_0, 0, E2_0, 0, 0, 0]

    # Time points for evaluation
    t_eval = np.arange(0, T + dt, dt)

    # Solve the system
    sol = solve_ivp(seir_model, [0, T], u0, t_eval=t_eval, method='RK45')
    return sol.t, sol.y

def plot_SEIR(t, u):
    labels = ["Susceptible", "Exposed (E1)", "Exposed (E2)", "Infected", "Recovered", "Deaths"]
    for i, label in enumerate(labels):
        plt.plot(t, u[i], label=label)
    plt.xlabel("Days since 15.02.2020")
    plt.ylabel("Population")
    plt.legend()
    plt.title("SEIR Model with Imported Infections")
    plt.show()

if __name__ == "__main__":
    # Initialize Beta from Problem 4
    beta = Beta('beta_values.txt')

    # Solve the SEIR model with import of infections
    t, u = solve_SEIRimport(T=1000, dt=1, S_0=5.5e6, E2_0=100, beta=beta, sigma=10)

    # Plot the results
    plot_SEIR(t, u)
