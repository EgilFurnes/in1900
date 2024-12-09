# 
# 
# Egil Furnes 
# egilsf@uio.no
# Final project in IN1900, fall 2024
# 
#

# Problem 4

# importerer pakker
import numpy as np
import matplotlib.pyplot as plt
# importerer også date og timedelta fra datetime
from datetime import date, timedelta

# definerer class Beta
class Beta:

    # definerer __init__ 
    def __init__(self, filename):
        self.beta = []
        self.t_start = []
        
        # leser inn tekstfilen beta_values
        # hentet fra prosjektbeskrivelsen
        with open('beta_values.txt') as infile:

            # hopper over et par tomme linjer
            infile.readline()  
            infile.readline()

            # leser inn linjene i filen hver for seg
            for line in infile:
                start_date, _, beta_value = line.split()
                self.beta.append(float(beta_value))
                self.t_start.append(self.str2date(start_date))

        self.t_start_days = [(t - self.t_start[0]).days for t in self.t_start]

    # omgjør strings til datoer for hver linje
    def str2date(self, date_str):
        # splitter opp i dag, måned og år
        day, month, year = (int(n) for n in date_str.split('.'))
        # returnerer verdier for år, måned og dag
        return date(year, month, day)

    # definerer __call__ for funksjonen
    def __call__(self, t):
        for i in range(len(self.t_start_days) - 1):
            if self.t_start_days[i] <= t < self.t_start_days[i + 1]:
                return self.beta[i]
        return self.beta[-1]

    # definerer en plottefunksjon for verdier av beta
    def plot(self, T):
        t = np.linspace(0, T, 1000)
        # henter ut verdiene fra beta
        b_values = np.vectorize(self.__call__)(t)
        # plotter betaverdier over tid
        plt.step(t, b_values, where='post')
        plt.xlabel("days since 15.02.2020")
        plt.ylabel("beta value")
        plt.title("beta/time")
        plt.show()


# initierer betafunksjonen og følgende funksjoner 
# for solve_SEIR og plot_SEIR
if __name__ == "__main__":
    from SEIR import solve_SEIR, plot_SEIR
    beta = Beta('beta_values.txt')
    beta.plot(1000)
    t, u = solve_SEIR(T=1000, S_0=5.5e6, E2_0=100, beta=beta, dt=1.0)
    plot_SEIR(t, u)

"""Kjøreeksempel:"""
"""test_SEIR_beta_const passed with a toleranse of 1e-10"""
"""test_SEIR_beta_var passed with a tolerance of 1e-10!"""
"""viser beta over tid som i prosjektbeskrivelsen"""
"""viser et relativt kjedelig SEIR plot med lite endringer"""
