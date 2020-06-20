import sys
from os import listdir
import numpy as np
import pandas as pd


input_string = list(sys.argv)


def quantiles_random_cat(path):
    order = np.arange(256)
    np.random.shuffle(order)
    order = list(order)

    catList = listdir(path)
    weights = []
    number = 0
    for cat in catList:
        deepPath = path+cat+"/"
        imageList = listdir(deepPath)
        place = order.index(number)
        quantile = -1
        for i in range(6):
            if place < 26 * (i+1):
                quantile = i
                break
        if quantile == -1:
            if place < 181:
                quantile = 6
            elif place < 206:
                quantile = 7
            elif place < 231:
                quantile =8
            else:
                quantile =9

        for images in imageList:
            weights.append(pow(float(input_string[2]),quantile))
        number += 1
    return weights

# To create without use of bins
def random_cat_weights(path):
    order = np.arange(256)
    np.random.shuffle(order)
    order = list(order)
    print(order)
    catList = listdir(path)
    weights = []
    number = 0
    for cat in catList:
        deepPath = path+cat+"/"
        imageList = listdir(deepPath)
        place = order.index(number)
        for images in imageList:
            weights.append(pow(  pow(float(input_string[2]),6/85)  ,place))
        number += 1
    return weights

path = input_string[1]
weights = pd.DataFrame(quantiles_random_cat(path), columns = ['Weight'])
total = weights['Weight'].sum()
weights = weights.multiply(other = 1 / total)

# checking all went well
check_sum = 0
count = 0
m = 0
for a in range(np.size(weights,0)):
    check_sum += weights.iloc[a,0]
    if weights.iloc[a,0] > 0:
        count += 1
        if weights.iloc[a,0] > m:
            m = weights.iloc[a,0]
print("check sum value is:" + str(check_sum))
print("Number of nonzero is " + str(count))
print("max weight is " + str(m))

weights.to_csv(input_string[3])