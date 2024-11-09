
# uke 2

# langtangen: 2.1 f2c_table_while 

print(f"{'Fahrenheit':>10} | {'Celsius':>7}")
print("-" * 30)

F = 0
while F<101:
    C = (5/9)*(F-32)

    print(f"{F:10d} | {C:7.2f}C")
    F = F+10
print(f"\n"*5)    

# langtangen: 2.3 primes

primes = [2,3,5,7,11,13]
p = 17
primes.append(p)
for i in primes:
    print(i)
print(f"\n"*2)

# oppgaveheftet: 2.2 interest_rate

def interest_rate(P, n, r):
    A = P*(1+r/100)**n
    print(f"An initial ${P:.2f} with {r}% interest over {n} years,")
    print(f"will grow to ${A:.2f}.")

interest_rate(P=1000, r=5, n=3)
print(f"\n"*2)

# oppgaveheftet: 2.3 population

from math import *

def population(B, k, t):
    C = ((B/5000)-1)/exp(-k*0)
    print(C)
    Nt = B/(1+C*exp(-k*t))
    print(f"the population after 24 hours is {Nt:.2f}")

population(B=50000, k=0.2, t=24)
print(f"\n"*2)

# oppgaveheftet: 2.4 find_roots

def roots(a, b, c):
    x1 = (-b+sqrt(b**2-4*a*c))/(2*a)
    x2 = (-b-sqrt(b**2-4*a*c))/(2*a)
    print(f"The roots for the equation")
    print(f"{a}x^2+{b}x+{c}=0 are")
    print(f"{x1:.2f} and {x2:.2f}")

roots(6, -5, 1)

# oppgaveheftet: 2.5 hydrogen

def hydrogen(r):
    ke = 9*10**9
    e = 1.6*10**(-19)
    G = 6.7*10**(-11)
    mp = 1.7*10**(-27)
    me = 9.1*10**(-31)
    Fc = ke*e**2/r**2
    Fg = G*mp*me/r**2
    print(f"Coloumb: {Fc}, Gravitation: {Fg}")
    print(f"Ratio of {Fc/Fg}")

print(f"\n"*2)
hydrogen(r=5.3*10**(-11))

# oppgaveheftet: 2.6 formulas_shapes

from math import pi
h = 5.0
b = 2.0
r = 1.5

para = h*b
print(f"parallelogram {para:.3f}")
square = b**2
print(f"square {square:g}")
circle = pi*r**2
print(f"circle {circle:.3f}")
cone = 1.0/3*pi*r**2*h
print(f"cone {cone:.3f}")

# oppgaveheftet: 3.1 multiplication

print(f"while:")
i = 1
while i<11:
    print(5*i)
    i+=1

print("\n")
print(f"for:")
for i in range(11):
    print(5*i)
    i+=1

# langtangen: 2.2 fc2_approx_table

print(f"\n")
F = 0
while F<101:
    F = F
    C = (5/9)*(F-32)
    Ch = (F-30)/2
    print(f"{F:5.1f} {C:5.1f} {Ch:5.1f}")
    F+=10

# langtangen: 2.4 odd

print(f"\nodd:")
def odd(n):
    i = 1
    while i<n+1:
        print(i)
        i+=2

odd(100)

# fysikk: 2.2 relativistic_momentum

def moment(m, v0, v1):
    c = 3*10**8
    while v0 <= v1:
        v = v0*c
        p = m*v
        y = 1/(sqrt(1-(v**2/c**2)))
        pr = m*v*y
        print(f"{v0:10.1f}c | {p:10.2e} | {pr:10.2e}")
        v0 += 0.1

moment(1200, 0, 0.9)

