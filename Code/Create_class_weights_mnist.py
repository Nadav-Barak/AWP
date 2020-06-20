import numpy as np
import pandas as pd
import sys

input_string = list(sys.argv)


data = np.genfromtxt(input_string[1], delimiter=',' , usecols = (0,1))
order = np.arange(10)
np.random.shuffle(order)
order = list(order)

weights = []

for i in range(np.size(data,0)):
# change number here to change exp factor
    weights.append(pow(float(input_string[2]),order.index(data[i,0])))

weights = pd.DataFrame(weights, columns = ['Weight'])
total = weights['Weight'].sum()

weights = weights.multiply(other = 1 / total)

# checking all is good
check_sum = 0
count = 0
m = np.inf
for a in range(np.size(weights,0)):
    check_sum += weights.iloc[a,0]
    if weights.iloc[a,0] > 0:
        count += 1
        if weights.iloc[a,0] < m:
            m = weights.iloc[a,0]
print("check sum value is:" + str(check_sum))
print("Number of nonzero is " + str(count))
print("min weight is " + str(m))

weights.to_csv(input_string[3])

