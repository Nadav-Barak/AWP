import numpy as np
from scipy.cluster.hierarchy import linkage
import pandas as pd
import sys

input_string = list(sys.argv)


data = np.genfromtxt(input_string[1] , delimiter=',')
data = data[1:, :] # remove row numbering for Caltech \ labels for MNIST
links = linkage(data, input_string[2])
df = pd.DataFrame(links)
df.to_csv(input_string[3])