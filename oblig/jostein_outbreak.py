"""Prosjektoppgave, Jostein Nermoen in1900 høst 2024"""
#Oppgave 3
import numpy as np 
from ODESolver import * 
from SEIR import * 
import matplotlib.pyplot as plt 

def beta(t): 
    if t <=30: 
        return 0.4 
    if t > 30: 
        return 0.083 

t,u = solve_SEIR(150, 1.0 , 5.5e6, 100, beta)
plot_SEIR(t,u)

"""KJøreeksempel: får opp et plott der grafene ser ut som rette linjer. tyder på at når beta blir mindre, altså smittefaren, 
så minker også veksten til kurvene, dvs mindre variasjoner mellom de forskjellige gruppene."""