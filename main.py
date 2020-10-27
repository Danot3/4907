# coding=utf-8
# This is a sample Python script.

import csv
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from pandas import DataFrame
from sklearn.cluster import KMeans




#intake data from csv file and set it to an array
data = np.genfromtxt('set_D2.csv', delimiter = ',')
print(data)
        



plt.plot(data)
plt.show()