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
    return 0.4 if t < 30 else 0.083 

t, u = solve_SEIR(150, 1.0 , 5.5e6, 100, betap)
plot_SEIR(t, u)

