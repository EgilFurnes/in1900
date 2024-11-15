# 
# 
# Egil Furnes 
# egilsf@uio.no
# Final project in IN1900, fall 2024
# 
#

# Problem 3

import numpy as np
import matplotlib.pyplot as plt 
from SEIR import solve_SEIR, plot_SEIR

def betap(t):
    if t < 30:
        return 0.4  # Before restrictions
    else:
        return 0.083  # After restrictions

T, dt, S_0, E2_0 = 150, 1.0, 5.5e6, 100

t, u = solve_SEIR(T, dt, S_0, E2_0, betap)
plot_SEIR(t, u)

smittetopp = max(u[:, 3])
print(f"Max number of infected people during the outbreak: {smittetopp:.2f}")
