import numpy as np

class Line:
    def __init__(self, c0, c1):
        self.c0, self.c1 = c0, c1

    def __call__(self, x):
        return self.c0 + self.c1*x
    
    def table(self, L, R, n):
        s = ''
        for x in np.linspace(L, R, n):
            y = self(x)
            s += f'{x:12g} {y:12g}\n'
        return s
    
class Parabola(Line):
    def __init__(self, c0, c1, c2):
        super().__init__(c0, c1)
        self.c2 = c2
    
    def __call__(self, x):
        return super().__call__(x) + self.c2*x**2
    
