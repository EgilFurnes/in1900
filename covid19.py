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
from SEIR import SEIR, solve_SEIR, plot_SEIR

class SEIRimport(SEIR):
    def __init__(self, beta=0.33, r_ia=0.1, 
                 r_e2=1.25, lmbda_1=0.33, lmbda_2=0.5, 
                 p_a=0.4, mu=0.2, sigma=10):
        super().__init__(beta=beta, r_ia=r_ia, r_e2=r_e2, 
                         lmbda_1=lmbda_1, lmbda_2=lmbda_2, 
                         p_a=p_a, mu=mu)
        self.sigma = sigma

    def __call__(self, t, u):
        derivatives = super().__call__(t, u)
        
        dS, dE1, dE2, dI, dIa, dR = derivatives
        
        dE2 = dE2 + self.sigma  

        return [dS, dE1, dE2, dI, dIa, dR]
    
def solve_SEIR(T, dt, S_0, E2_0, beta, sigma):
    model = SEIRimport(beta=beta, sigma=sigma)  # Use SEIRimport with sigma
    initial = [S_0, 0, E2_0, 0, 0, 0]
    timeval = np.arange(0, T + dt, dt)
    solution = solve_ivp(model, (0, T), initial, t_eval=timeval, method='RK45')
    t = solution.t
    u = solution.y.T
    return t, u

def betap(t):
    if t < 30:
        return 0.4  
    else:
        return 0.083  

T = 150  # Total time of simulation (in days)
dt = 1.0  # Time step (1 day)
S_0 = 5.5e6  # Initial susceptible population (e.g., 5.5 million)
E2_0 = 1000  # Initial exposed population in stage 2 (e.g., 1000)
sigma = 10  # Influx of infected people per day (Σ)

t, u = solve_SEIR(T, dt, S_0, E2_0, betap, sigma)

plot_SEIR(t, u)

smittetopp = max(u[:, 3])
print(f"Max number of infected people during the outbreak: {smittetopp:.2f}")
