import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# number of rows from ratings.dat file which will be processed
NUMBER_OF_ROWS = 100000

columns = ['UserID', 'MovieID', 'Rating']
# read data
data = pandas.read_csv('./data/ml-10M100K/ratings.dat', sep='::', engine='python', header=None, names=columns, usecols=columns, nrows=NUMBER_OF_ROWS)
data = data.pivot(index=columns[0], columns=columns[1], values=columns[2])  # transform data
data = data.fillna(0)  # replace missing values with zeros

# heat-map before DataFrame sort
sns.heatmap(data, yticklabels=False, xticklabels=False)
plt.savefig('./images/heatmap-before-sort.png')
plt.clf()

# Sort user-item DataFrame so that the density decreases along the main diagonal
a = data.values
a.sort(axis = 1)
a = a[:, ::-1]
a.sort(axis = 0)
a = a[::-1, :]
data = pandas.DataFrame(a, data.index, data.columns)
print data

# heat-map after DataFrame sort
sns.heatmap(data, yticklabels=False, xticklabels=False)
plt.savefig('./images/heatmap-after-sort.png')