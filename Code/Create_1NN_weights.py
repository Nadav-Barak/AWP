import numpy as np
import pandas as pd
from os import listdir
import cv2
import sys




def resize_and_scale(img, size, scale):
    img = cv2.resize(img, size)
    return scale - np.array(img, "int")

input_string = list(sys.argv)

source = np.genfromtxt(input_string[1], delimiter=',')
source = source[1:, 1:]
size = np.size(source, 0)
print("Number of rows in source is: " + str(size))
weights = np.zeros(size)

path = input_string[2]
print("path is " + path)

if len(input_string) == 4:
    relevent = []
else:
    relevent = input_string[4:]

counter = 0
sz = (224,224)
catList = listdir(path)
for cat in catList:
    deepPath = path+cat+"/"
    if len(relevent) == 0 or cat in relevent:
        imageList = listdir(deepPath)
        for images in imageList:
            if 'jpg' in images:
                img = cv2.imread(deepPath + images)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized_img = resize_and_scale(img, sz, 255)
                vec = resized_img.flatten()
                min_dis = np.inf
                location = -1
                for i in range(size):
                    distance = np.linalg.norm(source[i, :] - vec)
                    if distance < min_dis:
                        min_dis = distance
                        location = i
                weights[location] += 1
        print("Done with category " + cat)
        counter +=1

print("counted categories is " + str(counter))
#normalizing:
df = pd.DataFrame(weights, columns = ['Weight'])

total = df['Weight'].sum()
# weights['Weight'] = df['Weight'].apply(lambda x: pow(2,x) )

df = df.multiply(other = 1 / total)



# checking all is good
check_sum = 0
count = 0
m = 0
for a in range( len(df['Weight'])):
    check_sum += df.iloc[a,0]
    if df.iloc[a,0] > 0:
        count += 1
        if df.iloc[a,0] > m:
            m = df.iloc[a,0]
print("check sum value is:" + str(check_sum))
print("Number of nonzero is " + str(count))
print("max weight is " + str(m))

df.to_csv(input_string[3])
