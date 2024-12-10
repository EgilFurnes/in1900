

a = [4]*4
b = [0,1]+[0,1]
c = [2*i for i in range(1,4)]
d = [1]*4
e = [4]*1 +[4]*3
print(a,b,c,d,e)

for i in [a,b,c,d,e]:
    print(len(i))

def f(x,y,z,w=0):
    return x*y**z + w

# f(1,2,w=2)
# f(1,2,3)
# f(1,2,3,4)
# f(1,2,z=6)
# f(1,2)
# f(y=3,x=6,6)

x = [k**2 for k in range(1,100,3)]
# 1^2=1
# 4^2=16
# 7^2=49
# 10^2=100
y = x[3]-x[0]
print(y)

x = [2**k for k in range(11)]
# 2^0=1, 2^1=2, 2^2=4, 2^3=8 ...
# [1, 2, 4, 8, 16, 32, 64, 128]
y = [x[k+1]/x[k] for k in range(10)]
# 2+2+2+2+2+2+2+2+2+2=20
z = sum(y)
print(z)

x = 2
y = 4
def f(x,y):
    z = x*y
    return z
result = f(x,x)
print(result)

def poly(t,a,b,c):
    p = a*t**2 + b*t + c
    return p

print(poly(10,5,2,1))

def harmonic(n):
    csum = 0
    for i in range(1, n+1):
        ledd = 1/i
        csum += ledd 
    return csum 

print(harmonic(2))

def harmonic2(n):
    csum = 0
    for i in range(1, n+1):
        ledd = 1/i
        csum += ledd
    return csum, 1/(n+1)

print(harmonic2(100))


def is_prime(k):
    for i in range(2, k):
        if k % i == 0:
            print(True)
        else:
            print(False)

# print(is_prime(10))

def is_prime(k):
    if k<=1:
        return False
    for i in range(2, k):
        if k % i == 0:
            return False
    return True

print(is_prime(11))

def primes(n):
    primtall = []
    num = 2
    while len(primtall) < n:
        if is_prime(num):
            primtall.append(num)
        num += 1
    return primtall

print(primes(10))




from math import *
y = sin(x)*log(x)

C = [5, 10, 40, 45]
C.append(50)
print(C)

def amount(n):
    P = 100
    r = 5.0
    return P*(1+r/100)**n

year1 = 10
a1 = amount(year1)
a2 = amount(5)
print(a1, a2)
print(amount(6))
a_list = [amount(year) for year in range(11)]
print(a_list)

def amount(P, r, n):
    return P*(1+r/100.0)**n

a1 = amount(100, 5.0, 10)
a2 = amount(10, r=3.0, n=6)
a3 = amount(r=4, n=2, P=100)

print(a1, a2, a3)

def yfunc(t, v0):
    g = 9.81
    y = v0*t-0.5*g*t**2
    dydt = v0-g*t
    return y, dydt

print(f"\n")
print(yfunc(10, 100))
position, velocity = yfunc(0.6, 3)
print(position, velocity)
print(position)
print(velocity)


def f(x):
    return x, x**2, x**4

s = f(2)
print(type(s), s)
x, x2, x4 = s
print(s)
print(x, x2, x4)

def L(x,n):
    s = 0
    for i in range(1, n+1):
        s += x**i/i
    return s 

x = 0.5
from math import log
print(L(x,3), L(x, 10), -log(1-x))

from math import log
def L2(x, n):
    s = 0
    for i in range(1, n+1):
        s += x**i/i
    value_of_sum = s

    error = -log(1-x) - value_of_sum
    return value_of_sum, error 

x = 0.8; n = 10
value, error = L2(x, n)
print(value, error)
print(f"\n")

def table(x):
    print(f'x={x}, -ln(1-x)={-log(1-x)}')
    for n in [1,2,10,100]:
        value, error = L2(x,n)
        print(f"n={n:4d} approx: {value:7.6f}, error: {error:7.6f}")

table(0.5)

from math import sin, pi
def f(x):
    if 0 <= x <= pi:
        return sin(x)
    else:
        return 0
print(f(2))

def N(x):
    if x < 0:
        return 0
    elif 0 <= x < 1:
        return x
    elif 1 <= x < 2:
        return 2-x
    elif x >= 2:
        return 0
    
print(N(1))

def f(x): return (sin(x) if 0 <= x <= pi else 0)
print(f(3))

def d2df(f, x, h=1E-6):
    r = (f(x-h)-2*f(x)+f(x+h))/float(h*h)
    return r
# print(d2df(x**2,1))

def f(x):
    return x**2 - 1

f = lambda x: x**2-1

df2 = d2df(lambda x: x**2-1, 1.5)
print(df2)

from math import exp
def bisection(f, a, b, tol=1e-3):
    if f(a)*f(b) > 0:
        print(f"No roots or more than one root in [{a},{b}]")
        return
    m = (a+b)/2
    while abs(f(m))>tol:
        if f(a)*f(m) < 0:
            b = m
        else:
            a = m
        m = (a+b)/2
    return m

f = lambda x: x**2-4*x+exp(-x)
sol = bisection(f, -0.5, 1, 1e-6)
print(f"x = {sol:g} is an approx root, f({sol:g}) = {f(sol):g}")

from math import exp
def Newton(f, dfdx, x0, tol=1e-3):
    f0 = f(x0)
    while abs(f0) > tol:
        x1 = x0-f0/dfdx(x0)
        x0 = x1
        f0 = f(x0)
    return x0

f = lambda x: x**2-4*x+exp(-x)
dfdx = lambda x: 2*x-4-exp(-x)
sol = Newton(f,dfdx,0,1e-6)

print(f"x = {sol:g} is an approx root, f({sol:g}) = {f(sol):g}")

def double(x):
    return 2*x

def test_double():
    x = 4
    expected = 8
    computed = double(x)
    success = computed == expected
    msg = f"computed {computed}, expected {expected}"
    assert success, msg 

print(test_double())

from math import sin, pi

def f(x):
    if 0 <= x <= pi:
        return sin(x)
    else: 
        return 0
    
def test_f():
    x1, exp1 = -1.0, 0.0
    x2, exp2 = pi/2, 1.0
    x3, exp3 = 3.5, 0.0 
    
    tol = 1e-10

    assert abs(f(x1)-exp1) < tol, f'Failed for x = {x1}'
    assert abs(f(x2)-exp2) < tol, f'Failed for x = {x2}'
    assert abs(f(x3)-exp3) < tol, f'Failed for x = {x3}'

print(test_f())

from math import exp
# h = input('Input the altitude (in meters): ')
# h = float(h)
# p0 = 100.0
# h0 = 8400
# p = p0*exp(-h/h0)
# print(p)

infile = open('sample_file.txt', 'r')
print(infile)

with open('sample_file.txt', 'r') as file:
    content = file.readlines()

for line in content:
    print(line.strip())

infile = open('number_file.txt', 'r')
mean = 0
lines = 0
for line in infile:
    number = float(line)
    mean = mean + number
    lines += 1
mean = mean/lines
print(f'the mean value is {mean}')

s = "This is a typical string"
csvline = "Excel;sheets;often;use;semicolon;as;separator"
print(s.split())
print(csvline.split())
print(csvline.split(';'))


months = []
values = []
for line in infile:
    words = line.split()
    months.append(words[0])
    values.append(float(words[1]))


# filename = "rainfall.txt"

def extract_data(filename):
    infile = open(filename, 'r')
    infile.readline()
    months = []
    rainfall = []
    for line in infile:
        words = line.split()
        months.append(words[0])
        rainfall.append(words[1])
    infile.close()
    months = months[:-1]
    annual_avg = rainfall[-1]
    rainfall = rainfall[:-1]
    return months, rainfall, annual_avg

months, values, avg = extract_data('rainfall.txt')
print(months, values, avg)

print('The average rainfall for the months:')
for month, value in zip(months, values):
    print(month, value)
print('The average rainfall ofr the year:', avg)

data = \
[[ 0.75, 0.29619813, -0.29619813, -0.75 ],
[ 0.29619813, 0.11697778, -0.11697778, -0.29619813],
[-0.29619813, -0.11697778, 0.11697778, 0.29619813],
[-0.75, -0.29619813, 0.29619813, 0.75 ]]

with open('tmp_table.txt', 'w') as outfile:
    for row in data:
        for column in row:
            outfile.write(f'{column:14.8f}')
        outfile.write('\n')

import sys
try:
    h = float(sys.argv[1])
except:
    print('You failed to provide a command line arg.!')
#     exit()

# p0 = 100.0; h0 = 8400
# print(p0*exp(-h/h0))

print("hello")

from interest import years, babes 
P = 1; r = 5
n = years(P,2*P,P)
print(f'money has doubled after {n} years')

if __name__ == '__main__':
    print("hello bitch")

print(__name__)
print(babes())
print(__name__)

if __name__ == 'main':
    A = 2.31525
    P = 2.0
    r = 5
    n = 3
    p = 5
    A_ = present_amount(P,r,n)
    P_ = initial_amount(A,r,n)
    n_ = years(P,A,r)
    r_ = annual_rate(P,A,n)
    print(f'A={A_} ({A}) P={P_} ({A}) n={n_} ({n}) r={r_} ({p})')


