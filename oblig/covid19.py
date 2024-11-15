"""Oblig 2, in1900 Jostein nermoen 2024"""
#Oppgave 5. 
from ODESolver import *
from SEIR import *
import matplotlib.pyplot as plt 
import numpy as np
from lockdown_riktig import Beta
from datetime import date, timedelta 


class SEIRimport(SEIR):

    def __init__(self, beta, sigma):   #like mange argumenter
        super().__init__(beta)
        self.sigma = sigma

    def __call__(self, t, u):
        l = super().__call__(t,u)           #Gjenbruk
        l[2] = l[2] + self.sigma 
        return l 

def solve_SEIRimport(T, dt , S_0 ,E2_0, beta,sigma):
    funksjon = SEIRimport(beta, sigma)
    N = round(T/dt)
    method = RungeKutta4(funksjon)
    method.set_initial_condition([S_0, 0, E2_0, 0, 0, 0])
    t,u = method.solve((0,T), N)
    return t, u 


beta = Beta('beta_values.txt')
t, u = solve_SEIRimport(T=1000, dt = 1, S_0=5.5e6, E2_0=100, beta = beta,sigma=10)
plot_SEIR(t, u) 

""""Kjøreeksempel: dukker opp et plot, men vi ser at det er større vekst blant den ene gruppen. Så, mer variasjon enn i plottet vi fikk i eksempel fire."""
