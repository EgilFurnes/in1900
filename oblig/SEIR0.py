'''Prosjektoppgave, Jostein Nermoen in1900 høst 2024, 
problem én''' 
from ODESolver import *
import numpy as np
import matplotlib.pyplot as plt



class SEI0:
    def __init__(self, beta = 0.33, r_ia = 0.1,
                r_e2 = 1.25, lmbda_1 = 0.33,
                lmbda_2 = 0.5, p_a=0.4, mu = 0.2):
        self.beta = beta
        self.r_ia = r_ia
        self.r_e2 = r_e2
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2
        self.p_a = p_a
        self.mu = mu

    def __call__(self,t,u):        #S: suspectible, I, infected, r, du vet 
        beta = self.beta
        r_ia = self.r_ia
        r_e2 = self.r_e2
        lmbda_1 = self.lmbda_1
        lmbda_2 = self.lmbda_2
        p_a = self.p_a
        mu = self.mu
    
        S,E1,E2,I,Ia,R = u
        N = sum(u)
        dS = -beta*S*I /N - r_ia*beta*S*Ia/N\
            -r_e2*beta*S*E2/N

        dE1 = beta*S*I /N+r_ia*beta*S*Ia/N\
        +r_e2*beta*S*E2/N-lmbda_1*E1

        dE2 = lmbda_1*(1-p_a)*E1-lmbda_2*E2
        dI = lmbda_2*E2-mu*I
        dIa = lmbda_1*p_a*E1 - mu*Ia
        dR = mu*(I+Ia)
        return [dS,dE1,dE2,dI,dIa,dR] 

#a), test funksjon
def test_seir0():
    expected = [-0.12925, -0.20075, -0.302,0.3,-0.068, 0.4]     #regnet ut for hånd
    
    sei0_test = SEI0()
    u = [1,1,1,1,1,1]       #testverdier
    t = 0
    calculated = sei0_test(t,u)

    tol = 1e-10
    tv = []                              #lager liste av sannhetsverdi til utsagnene
    for a,b in zip(expected,calculated):
        tv.append(abs(a - b) <= tol)
    success = tv == [True,True,True,True,True,True]
    assert success
    

test_seir0()


"""Kjøreeksempel: når programmet kjøres skjer ingenting"""
#b) 

def solve_SEIR(T, dt , S_0 ,E2_0, beta):
    funksjon = SEI0(beta=beta)
    N = round(T/dt)
    method = RungeKutta4(funksjon)
    method.set_initial_condition([S_0, 0, E2_0, 0, 0, 0])
    t,u = method.solve((0,T), N)
    return t, u

"""KJøreeksempel: her skjer heller ingenting (naturlig nok)"""
    
#c) plotte funksjon 

def plot_SEIR(t,u):
    for k in range(0,6):
        if k != 1 and k != 2:
            plt.plot(t, (u[:, k]))
    plt.xlabel("tid")
    plt.ylabel("funksjonsverdi")
    plt.legend(['S','I','Ia','R'])
    plt.show()
    
t, u = solve_SEIR(150, 1.0 , 5.5e6, 100, 0.4)
plot_SEIR(t,u)

"""Kjøreeksempel:  et passende plot dukker opp   """ 


