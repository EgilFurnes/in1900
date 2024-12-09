

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


