""""Jostein Nermoen, in1900"""
#Oppgave 4
import numpy as np
import matplotlib.pyplot as plt 
from datetime import date, timedelta 


class Beta:                              #skal implementere piecewise constant B, 
    def __init__(self, filename): 
        self.beta = []
        self.t = []                    #liste med datoer uformatert

        self.t_days_int = [0]           #setter initalverdi tid, lettere sånn                   #gitt intervall skalden bestemme verdi(som en funksjon)
        with open(filename) as infile:
            infile.readline()
            infile.readline()
            for lines in infile:
                line = lines.split()
                self.beta.append(line[2])
                self.t.append(line[0])

        for n in range(1,len(self.t)):
            delta = self.str2date(self.t[n]) - self.str2date(self.t[0])
            n_days = delta.days
            self.t_days_int.append(n_days)




    def str2date(self, date_str): 
        day, month, year = (int(n) for n in date_str.split('.'))
        return date(year,month,day) 


    def __call__(self, t):     #Skal ta array t som argument, løse for b: 
        if t>= self.t_days_int[-1]:     #setter at t over endepunket.
            beta = float(self.beta[-1])
        else : 
            for n in range(len(self.t_days_int)-1): 
                if self.t_days_int[n]<= t and t < self.t_days_int[n+1]: 
                    beta = float(self.beta[n])
                    
        return beta 
    

    def plot(self, T): 
        t = np.linspace(0, T, 1000)
        bs = []
        for e in t: 
            b = self(e)
            bs.append(b)
        

        plt.plot(t, np.array(bs))
        plt.show()




if __name__ == "__main__":
    from SEIR import *
    beta = Beta('beta_values.txt')
    beta.plot(1000)
    t, u = solve_SEIR(T=1000, dt = 1, S_0=5.5e6, E2_0=100, beta=beta)
    plot_SEIR(t, u)


"""Kjøreeksempel: dukker op plots som ligner.
Kommentar: denne løsningen ligner veldig, ser det samme tilnærmet rette strekene som tidligere."""


