import sys
from os import listdir
import numpy as np
import pandas as pd
import cv2

def resize_and_scale(img, size, scale):
    img = cv2.resize(img, size)
    return 1 - np.array(img, "float32")/scale

def loadBatchImages(path):
    sz = (224,224)
    catList = listdir(path)
    caltech_data = []
    for cat in catList:
        deepPath = path+cat+"/"
        imageList = listdir(deepPath)
        for images in imageList:
            if "jpg" in images:
                img = cv2.imread(deepPath + images)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # change img to gray scale
                resized_img = resize_and_scale(img, sz, 255) # resize to a standard shape
                caltech_data.append(resized_img.flatten())
        print("Done with " + cat)
    caltech_data = np.array(caltech_data)
    return caltech_data


input_string = list(sys.argv)

path = input_string[1]
print("path is " + path)
df = pd.DataFrame(loadBatchImages(path))


df = df.multiply(other = 255)
df = df.round(0)
df.to_csv(input_string[2])