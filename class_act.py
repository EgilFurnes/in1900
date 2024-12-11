
from math import exp

def barometric(h, T):
    g = 9.81
    R = 8.314
    M = 0.02896
    p0 = 100.0

    return p0*exp(-M*g*h/(R*T))

print(barometric(1000, 40))

class Barometric:
    def __init__(self, T):
        self.T = T
        self.g = 9.81
        self.R = 8.314
        self.M = 0.02896
        self.p0 = 100.0

    def value(self, h):
        return self.p0 * exp(-self.M*self.g*h/(self.R*self.T))
    
b1 = Barometric(T=245)
p1 = b1.value(2469)
b2 = Barometric(T=273)
p2 = b2.value(2469)

print(b1, "\n", p1, "\n", b2, "\n", p2)

class Barometric1:
    def __init__(self, T):
        self.T = T

    def value(self, h):
        g = 9.81; R = 9.314; M = 0.02896; p0 = 100.0
        return p0 * exp(-M*g*h/(R*self.T))
    
class Baromertric2:
    def __init__(self, T):
        g = 9.81
        R = 8.314
        M = 0.02896
        self.h0 = R*T/(M*g)
        self.p0 = 100.0

    def value(self, h):
        return self.p0 * exp(-h/self.h0)
    
p1 = b1.value(2469)
print(p1)
p1 = Barometric.value(b1, 2469)
print(p1)

from math import sin, exp, pi
from numpy import linspace

def make_table(f, tstop, n):
    for t in linspace(0, tstop, n):
        print(t, f(t))

def g(t):
    return sin(t)*exp(-t)

make_table(g, 2*pi, 11)

b1 = Barometric(2469)
make_table(b1.value, 2*pi, 11)

print(f'\n')

class BankAccount:
    def __init__(self, first_name, last_name, number, balance):
        self.first_name = first_name
        self.last_name = last_name
        self.number = number
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        self.balance -= amount

    def print_info(self):
        first = self.first_name
        last = self.last_name
        number = self.number
        balance = self.balance
        s = f'{first} {last}, {number}, balance: {balance}'
        print(s)

a1 = BankAccount('John', 'Olsson', '19371554951', 2000)
print(a1)
a1.deposit(100)
a1.print_info()
a1.deposit(1234)
a1.print_info()
a1.withdraw(6969)
a1.print_info()

print("\n")

class Barometric:
    def __init__(self, T):
        self.T = T
        self.g = 9.81
        self.R = 8.314
        self.M = 0.02896
        self.p0 = 100.0
    
    def __call__(self, h):
        return self.p0 * exp(-self.M*self.g*h/(self.R*self.T))
    
    def __str__(self):
        return f'p0 * exp(-M*g*h/(R*T)'
    
    def __repr__(self):
        return f'Barometric({self.T})'

baro = Barometric(245)
p = baro(2346)
print(baro, p)
b = Barometric(245)
print(b)
print(b(2469))

print('\n')
print(b)
print(repr(b))
b = Barometric(271)
b2 = eval(repr(b))
print(repr(b2))


class A:
    def __init__(self, value):
        self.v = value

a = A(2)
print(dir(a))

a = A([1,2])
print(a.__dict__)
a.myvar = 10
print(a.__dict__)

# ---

class Derivative:
    def __init__(self, f, h=1E-5):
        self.f = f
        self.h = float(h)

    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x+h)-f(x))/h

from math import *

df = Derivative(sin)
x = pi

print(df(x))
print(cos(x))

def g(t): return t**3
dg = Derivative(g)
t = 1

print(dg(t))
print(f"{abs(dg(t)-3):g}")

def f(x): return 100000*(x-0.9)**2*(x-1.1)**3
dfdx = Derivative(f)
xstart = 1.01


class Polynomial:
    def __init__(self, coefficients):
        self.coeff = coefficients

    def __call__(self, x):
        s = 0
        for i in range(len(self.coeff)):
            s += self.coeff[i]*x**i
        return s
    
    def __add__(self, other):
        if len(self.coeff) > len(self.other):
            coeffsum = self.coeff[:]
            for i in range(len(other.coeff)):
                coeffsum[i] += other.coeff[i]
        else:
            coeffsum = other.coeff[:]
            for i in range(len(self.coeff)):
                coeffsum[i] += self.coeff[i]
        return Polynomial(coeffsum)
    
    def __mul__(self, other):
        M = len(self.coeff) - 1
        N = len(self.coeff) - 1
        coeff = [0]*(M+N+1)
        for i in range(0, M+1):
            for j in range(0, N+1):
                coeff[i+j] += self.coeff[i]*other.coeff[j]
        return Polynomial(coeff)
    

    
