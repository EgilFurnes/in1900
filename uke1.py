
import math as math 
import numpy as np 

# 1.3 seconds2years s43

def seconds2years(seconds):
    years = seconds/(60*60*24*365.25)
    print(f"{seconds} seconods computes to {years:2f} years!")

print(seconds2years(10**9))


# 1.4 length_conversion s43

def length_conversion(meters):
    inches = (meters/100)*2.54
    feet = inches/12
    yards = feet/3
    miles = yards/1760
    print(f"{meters:.2f} meters is equal to {inches:2f} inches, {feet:2f} feet, {yards:2f} yards, and {miles:2f} miles")

length_conversion(1000)

# 1.12 egg s46

def egg(Tw, Ty, M=67, rho=1.038, c=3.7, K=0.0054, T0=4):
    t = (M**(2/3)*c*rho**(1/3))/(K*np.pi**2*(4*np.pi/3)**(2/3))*math.log2(0.76*(T0-Tw)/(Ty-Tw))
    minutes = t/60
    print(f"The perfect egg of these proportions should be cooked {t:.2f} seconds!")
    print(f"Alternatively, {minutes:.2f} minutes")

egg(Tw=100, Ty=70)

