# 
# 
# Egil Furnes 
# egilsf@uio.no
# Final project in IN1900, fall 2024
# 
#

# Problem 3

# importerer pakker
import numpy as np
import matplotlib.pyplot as plt

# importerer også funksjonene solve_SEIR
# og plot_SEIR fra filen SEIR.py
from SEIR import solve_SEIR, plot_SEIR

# definerer en piecewise betafunksjon
# hvor beta er 0.4 for t<30
# hvor beta er 0.083 for t>30
def betap(t):
    if t < 30:
        return 0.4  # Before restrictions
    else:
        return 0.083  # After restrictions

# initierer verdier for T, dt, S_0, E2_0
T, dt, S_0, E2_0 = 150, 1.0, 5.5e6, 100

# løser for t og u ved å passere
# verdiene som argumenter gjennom solve_SEIR
t, u = solve_SEIR(T, dt, S_0, E2_0, betap)

# kjører funksjonen plot_SEIR for argumentene t og u
plot_SEIR(t, u)

# finner smittetoppen i løpet av perioden
smittetopp = max(u[:, 3])
print(f"Max number of infected people during the outbreak: {smittetopp:.2f}")

"""Kjøreeksempel"""
"""test_SEIR_beta_const passed with a toleranse of 1e-10"""
"""test_SEIR_beta_var passed with a tolerance of 1e-10!"""
"""Max number of infected people during the outbreak: 779.15"""
"""viser et plot som er ganske kjedelig, med tilsynelatende flate linjer"""