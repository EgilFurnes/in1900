

class Derivative: 
    def __init__(self, f, h=1E-5):
        self.f = f
        self.h = float(h)

    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x+h)-f(x))/h
    
from math import exp, sin, pi

def f(x): return exp(-x)*sin(4*pi*x)

dfdx = Derivative(f)

print(dfdx(1.2))

class Diff:
    def __init__(self, f, h=1E-5):
        self.f, self.h = f, h

class Forward1(Diff):
    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x+h)-f(x))/h
    
class Central2(Diff):
    def __call__(self, x):
        f, h = self.f, self.h
        return (f(x+h)-f(x-h))/(2*h)
    
class Central4(Diff):
    def __call__(self, x):
        f, h = self.f, self.h
        return (4./3)*(f(x+h)-f(x-h))/(2*h) - (1./3)*(f(x+2*h)-f(x-2*h))/(4*h)
    
from math import sin, pi
mycos = Central4(sin)
mycos(pi)

