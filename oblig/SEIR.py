"""Prosjektoppgave, Jostein Nermoen in1900 høst 2024""" 

#problem 2 
from ODESolver import *
import numpy as np 
import matplotlib.pyplot as plt

class SEIR:
    def __init__(self, beta, r_ia = 0.1,
                r_e2 = 1.25, lmbda_1 = 0.33,
                lmbda_2 = 0.5, p_a=0.4, mu = 0.2):
        if isinstance(beta, (float, int)): 
            self.beta = lambda t: beta 
        elif callable(beta):
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
        N = sum(u)                             #modifiserer beta slik at det er en funskjon som kalles på med den gitte t
        dS = -beta(t) *S*I /N - r_ia*beta(t)*S*Ia/N\
        -r_e2*beta(t)*S*E2/N

        dE1 = beta(t)*S*I /N+r_ia*beta(t)*S*Ia/N\
        +r_e2*beta(t)*S*E2/N-lmbda_1*E1

        dE2 = lmbda_1*(1-p_a)*E1-lmbda_2*E2
        dI = lmbda_2*E2-mu*I
        dIa = lmbda_1*p_a*E1 - mu*Ia
        dR = mu*(I+Ia)
        return [dS,dE1,dE2,dI,dIa,dR] 


"""Ikke noe kjøring enn så lenge"""
#b) 

def test_SEIR_beta_const(): 
    expected = [-0.12925, -0.20075, -0.302,0.3,-0.068, 0.4]   #Output fra forrige test
    SEIR_ = SEIR(0.33)
    t = 0 
    u = [1,1,1,1,1,1]
    calculated = SEIR_(t,u)
    tol = 1e-10
    tv = []
    for a,b in zip(expected,calculated):
        tv.append(abs(a - b) <= tol)
    success = tv == [True,True,True,True,True,True]
    assert success 

test_SEIR_beta_const()

#def test_SEIR_beta_const():    MÅ gjøres 


#c) 
def solve_SEIR(T, dt , S_0 ,E2_0, beta):
    funksjon = SEIR(beta=beta)
    N = round(T/dt)
    method = RungeKutta4(funksjon)
    method.set_initial_condition([S_0, 0, E2_0, 0, 0, 0])
    t,u = method.solve((0,T), N)
    return t, u

#d) 

def plot_SEIR(t,u, components=['S','I','Ia','R']):
    dicty = {'S': u[:,0], 'E1':u[:,1],'E2': u[:,2],'I':u[:,3],'Ia':u[:,4],'R':u[:,5]}
    
    for letter in components: 
        plt.plot(t,dicty[letter])
    
    plt.xlabel("tid")
    plt.ylabel("funksjonsverdi")
    plt.legend(['S','I','Ia','R'])
    plt.show()

#under ser man et eksempel som funker som det skal
"""t, u = solve_SEIR(150, 1.0 , 5.5e6, 100, 0.4)    
plot_SEIR(t,u,['S'])"""

"""kjøreeksempel: når jeg kjører med samme verdier som sist dukker det opp et plott som ser helt likt ut, 
testet også med components=['S'], og fikk riktig utfall"""

#d) 
if __name__ == "__main__":
    t, u = solve_SEIR(T=300, dt = 1.0, S_0=5.5e6, E2_0=100,beta=0.4)    #mangler dt? La til noe passende
    plot_SEIR(t, u)
    print(f'høyeste antal smittetede ved et tidspunkt i intervallet er {round(max(u[:,3]))}')

"""Kjøreeksempel: 'høyeste antal smittetede ved et tidspunkt i intervallet er 258544'
Kommentarer: med litt grov regning ser vi at 25000 * 0.05 er 1250, altså nesten dobbelt så mange
som det var/er tilgjengelige ventilatorer! """
