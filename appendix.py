
import numpy as np
import matplotlib.pyplot as plt

x0 = 100
p = 5
N = 4
index_set = range(N+1)
x = np.zeros(len(index_set))

x[0] = x0
for n in index_set[1:]:
    x[n] = x[n-1] + (p/100.0) * x[n-1]

plt.plot(index_set, x, 'ro')
plt.xlabel('years')
plt.ylabel('amount')
# plt.show()


x_old = x0
for n in index_set[1:]:
    x_new = x_old + (p/100.) * x_old
    x_old = x_new
print('final amount:', x_new)

x = x0
for n in index_set[1:]:
    x = x+(p/100.)*x
print(f'final amount: {x}')

import datetime

date1 = datetime.date(2017, 9, 29)
date2 = datetime.date(2018, 8, 4)
diff = date2 - date1
print(diff.days)

import numpy as np
import matplotlib.pyplot as plt
import datetime

x0 = 100
p = 5
r = p/360.0

date1 = datetime.date(2017, 9, 29)
date2 = datetime.date(2018, 8, 4)
diff = date2 - date1
N = diff.days
index_set = range(N+1)
x = np.zeros(len(index_set))

x[0] = x0
for n in index_set[1:]:
    x[n] = x[n-1] + (r/100.0) * x[n-1]

plt.plot(index_set, x)
plt.xlabel('days')
plt.ylabel('amount')
# plt.show()

p = np.zeros(len(index_set))
r = p/360.0
x = np.zeros(len(index_set))

x[0] = x0
for n in index_set[1:]:
    x[n] = x[n-1] + (r[n-1]/100.0) * x[n-1]

date0 = datetime.date(2017,9,29)
date1 = datetime.date(2018,2,6)
date2 = datetime.date(2018,8,4)
Np = (date1-date0).days
N = (date2-date1).days

p = np.zeros(len(index_set))
p[:Np] = 4.0
p[Np:] = 5.0
print(p)

