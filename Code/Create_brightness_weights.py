import numpy as np
import pandas as pd
import sys

input_string = list(sys.argv)


brightness = []
for gm_chunk in pd.read_csv(input_string[1],chunksize=50):
    print(gm_chunk.shape[1])
    for i in range(gm_chunk.shape[0]):
        brightness = brightness + [np.average(list(gm_chunk.iloc[i, 1:]))] # the first column is row numbering in Caltech/ label in MNIST.

print("Done data read")

quentiles = pd.DataFrame(list(pd.qcut(brightness, 10, labels=False)), columns = ['Weight'])
quentiles['Weight'] = quentiles['Weight'].apply(lambda x: pow(float(input_string[2]),x) )

print(quentiles.shape)
print(quentiles.head())
total = quentiles['Weight'].sum()
# weights['Weight'] = quentiles['Weight'].apply(lambda x: pow(2,x) )

quentiles = quentiles.multiply(other = 1 / total)

# checking all is good
check_sum = 0
count = 0
m = 0
for a in range( len(quentiles['Weight'])):
    check_sum += quentiles.iloc[a,0]
    if quentiles.iloc[a,0] > 0:
        count += 1
        if quentiles.iloc[a,0] > m:
            m = quentiles.iloc[a,0]
print("check sum value is:" + str(check_sum))
print("Number of nonzero is " + str(count))
print("max weight is " + str(m))

quentiles.to_csv(input_string[3])