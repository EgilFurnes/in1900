
temps = {'Oslo': 13, 'London': 15.4, 'Paris': 17.5, 'Madrid': 26}
for city in temps:
    print(f'The {city} temperature is temps{city}')

for city in sorted(temps):
    value = temps[city]
    print(value)


with open('deg2.txt', 'r') as infile:
    temps = {}
    for line in infile:
        city, temp = line.split()
        city = city[:-1]
        temps[city] = float(temp)

print(infile)

s = 'this is a string'
s.split()
print(s.split())

s = 'Berlin: 18.4 C at 4 pm'
print(s[0], s[1], s[-1], s[8:])
print(s.find('Berlin'))
print(s.find('18'))
print(s.find('Oslo'))

print('Berlin' in s)

if 'C' in s:
    print('C found')
else:
    print('no C')

print(s.split(':'))

strings = ['Newton', 'Secant', 'Bisection']
print(', '.join(strings))
print('  '.join(strings))

l1 = 'Oslo; 84 C at 5 pm'
words = l1.split()
l2 = ' '.join(words)
l1 == l2
print(l1 == l2)

s = '       text with leading/trailing space      \n'
print(s.strip())
print(s.lstrip())
print(s.rstrip())

s = 'Berlin: 18.4 C at 4 pm'
print('214'.isdigit())
print('214'.isspace())
print('214'.isdigit())
print('214'.isdigit())
print(s.startswith('Berlin'))
print(s.startswith('B'))
print(s.lower())
print(s.upper())

pairs = []
with open('pairs.txt', 'r') as lines:
    print(lines)
    for line in lines:
        words = line.split()
        print(words)
        for word in words:
            print(word)
            word = word[1:-1]
            n1, n2 = word.split(',')
            n1 = float(n1)
            n2 = float(n2)
            print(n1, n2)
            pair = (n1, n2)
            print(pair)
            pairs.append(pair)

print(pairs)


















