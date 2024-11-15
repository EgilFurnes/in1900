# 
# 
# Egil Furnes 
# egilsf@uio.no
# Final project in IN1900, fall 2024
# 
#

# Problem 4

import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

class Beta:
    def __init__(self, filename):
        self.beta = []
        self.t_start = []
        
        with open(filename) as infile:
            infile.readline()  
            infile.readline()
            for line in infile:
                start_date, _, beta_value = line.split()
                self.beta.append(float(beta_value))
                self.t_start.append(self.str2date(start_date))

        self.t_start_days = [(t - self.t_start[0]).days for t in self.t_start]

    def str2date(self, date_str):
        day, month, year = (int(n) for n in date_str.split('.'))
        return date(year, month, day)

    def __call__(self, t):
        for i in range(len(self.t_start_days) - 1):
            if self.t_start_days[i] <= t < self.t_start_days[i + 1]:
                return self.beta[i]
        return self.beta[-1]

    def plot(self, T):
        t = np.linspace(0, T, 1000)
        b_values = np.vectorize(self.__call__)(t)
        plt.step(t, b_values, where='post')
        plt.xlabel("days since 15.02.2020")
        plt.ylabel("beta value")
        plt.title("beta/time")
        plt.show()

if __name__ == "__main__":
    from SEIR import solve_SEIR, plot_SEIR
    beta = Beta('beta_values.txt')
    beta.plot(1000)
    t, u = solve_SEIR(T=1000, S_0=5.5e6, E2_0=100, beta=beta)
    plot_SEIR(t, u)

