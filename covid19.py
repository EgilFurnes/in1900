# 
# 
# Egil Furnes 
# egilsf@uio.no
# Final project in IN1900, fall 2024
# 
#

# Problem 5

# importerer pakker
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# importerer også funksjonene SEIR, solve_SEIR og plot_SEIR fra
# filen SEIR.py
from SEIR import SEIR, solve_SEIR, plot_SEIR

# initierer klassen SEIRimport 
class SEIRimport(SEIR):

    # definerer __init__
    # denne gangen inkluderer vi argumentet sigma, 
    # med defaultverdi 10
    def __init__(self, beta=0.33, r_ia=0.1, 
                 r_e2=1.25, lmbda_1=0.33, lmbda_2=0.5, 
                 p_a=0.4, mu=0.2, sigma=10):
        super().__init__(beta=beta, r_ia=r_ia, r_e2=r_e2, 
                         lmbda_1=lmbda_1, lmbda_2=lmbda_2, 
                         p_a=p_a, mu=mu)
        self.sigma = sigma

    # definerere __call__
    def __call__(self, t, u):
        derivatives = super().__call__(t, u)
        
        # definerer derivatene til S, E1, E2, 
        # I, Ia og R
        dS, dE1, dE2, dI, dIa, dR = derivatives
        dE2 = dE2 + self.sigma  

        # returnerer de deriverte
        return [dS, dE1, dE2, dI, dIa, dR]
    
# definerer en funksjon for å løse SEIR for T, dt, S_0, E2_0, beta og sigma
def solve_SEIR(T, dt, S_0, E2_0, beta, sigma):
    # benytter SEIRimport-funksjonen med sigma
    model = SEIRimport(beta=beta, sigma=sigma)  
    # setter intervall som tidligere for verdier og tid
    initial = [S_0, 0, E2_0, 0, 0, 0]
    timeval = np.arange(0, T + dt, dt)
    # løser ved bruk av solve_ivp fra scipy
    solution = solve_ivp(model, (0, T), initial, t_eval=timeval, method='RK45')
    # finner de løste verdiene for t og u
    t = solution.t
    u = solution.y.T
    # returnerer verdier for t og u
    return t, u

# definerer piecewise funksjon for beta
def betap(t):
    if t < 30:
        return 0.4  
    else:
        return 0.083  

# setter verdier for argumentene T, dt, S_0, E2_0 og sigma
T = 150  
dt = 1.0  
S_0 = 5.5e6  
E2_0 = 1000  
sigma = 10 

# løser for t og u ved bruk av solve_SEIR med piecewise beta og sigma
t, u = solve_SEIR(T, dt, S_0, E2_0, betap, sigma)

# plotter SEIR for t og u
plot_SEIR(t, u)

# finner smittetoppen
smittetopp = max(u[:, 3])
print(f"Max number of infected people during the outbreak: {smittetopp:.2f}")

"""Kjøreeksempel:"""
"""test_SEIR_beta_const passed with a toleranse of 1e-10"""
"""test_SEIR_beta_var passed with a tolerance of 1e-10!"""
"""viser SEIR plot - fortsatt relaltivt få endringer"""