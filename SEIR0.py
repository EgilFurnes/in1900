# 
# 
# Egil Furnes 
# egilsf@uio.no
# Final project in IN1900, fall 2024
# 
#

# Problem 1 

# importerer pakker
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# definerer class SEIR0
class SEIR0:
    # deferer __init__
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

    # definerer __call__ 
    def __call__(self, t, u):
        beta = self.beta
        r_ia = self.r_ia
        r_e2 = self.r_e2
        lmbda_1 = self.lmbda_1
        lmbda_2 = self.lmbda_2
        p_a = self.p_a
        mu = self.mu

        # verdier for S, E1, S2, I, Ia, R i u
        S, E1, E2, I, Ia, R = u
        N = sum(u)

        # tar derivatet av verdiene
        # S, E1, E2, I, Ia, and R
        dS = -beta * S * I / N - r_ia * beta * S * Ia / N \
            - r_e2 * beta * S * E2 / N
        dE1 = beta * S * I / N + r_ia * beta * S * Ia / N \
            + r_e2 * beta * S * E2 / N - lmbda_1 * E1
        dE2 = lmbda_1 * (1 - p_a) * E1 - lmbda_2 * E2
        dI = lmbda_2 * E2 - mu * I
        dIa = lmbda_1 * p_a * E1 - mu * Ia
        dR = mu * (I + Ia)

        # returnerer derivatene 
        return [dS, dE1, dE2, dI, dIa, dR]

# a)

# definerer en testfunksjon for SEIR0
def test_SEIR0():

    # benytter SEIR0-modellen
    model = SEIR0()

    # setter tid t til 0 og initialiserer u med 1'ere 
    t = 0
    u = [1,1,1,1,1,1]

    # output finnes ved å passere t og u som argumenter
    # i SEIR0-modellen
    output = model(t,u)

    # finner verdier for beta, r_ia, and r_e2
    beta, r_ia, r_e2 = model.beta, model.r_ia, model.r_e2

    # finner verdier for lmbda_1, lmbda_2, p_a, and mu
    lmbda_1, lmbda_2, p_a, mu = model.lmbda_1, model.lmbda_2, model.p_a, model.mu

    # setter inn S, E1, E2, I, Ia, og R i u
    S, E1, E2, I, Ia, R = u

    # kalkulerer sum av tuppelen u 
    N = sum(u)

    # forventede verdier for derivatene til modellen 
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

    # setter toleranse til 0.0000000001
    tol = 1e-10
    
    # undersøker om computed verdier er lavere
    # enn den tolererte feilmarginen
    for i, (computed, exp) in enumerate(zip(output, expected)):
        assert abs(computed-exp)<tol 

    # printing hvis testen passerer
    print(f"testene passerte med en margin på {tol}")

test_SEIR0()

"""Kjøreeksempel: """
"""testene passerte med en margin på 1e-10"""

# b)

# definerer funksjonene solve_SEIR
def solve_SEIR(T,dt,S_0,E2_0,beta):
    # benytter modellen SEIR0
    model = SEIR0(beta=beta)
    # initielle verdier med S_0 og E2_0, ellers stilt til 0
    initial = [S_0, 0, E2_0, 0, 0, 0]
    timeval = np.arange(0, T+dt, dt)
    # benytter solve_ivp fra scipy.integrate
    solution = solve_ivp(model, (0, T), initial, t_eval=timeval if dt else None, method='RK45')
    # finner de løste verdiene
    t = solution.t
    u = solution.y.T
    # returnerer verdier for t og u
    return t, u

"""Kjøreeksempel: """
"""ingen output"""

# c)

# definerer en plottefunksjon for SEIR
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
    # viser plottet
    plt.show()
    
# initierer verdier for T, dt, S_0, E2_0, beta
T, dt, S_0, E2_0, beta = 150, 1.0, 5.5e6, 100, 0.4
# passerer verdiene gjennom funksjonen solve_SEIR
t, u = solve_SEIR(T, dt, S_0, E2_0, beta)
# plotter SEIR
plot_SEIR(t, u)

"""Kjøreeksempel: """
"""viser the plot lignende det i prosjektbeskrivelsen"""