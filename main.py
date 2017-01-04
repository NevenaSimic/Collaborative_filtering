import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import model_based
import memory_based

# number of rows from ratings.dat file which will be processed
NUMBER_OF_ROWS = 10000

columns = ['UserID', 'MovieID', 'Rating']
# read data
data = pandas.read_csv('./data/ml-10M100K/ratings.dat', sep='::', engine='python', header=None, names=columns, usecols=columns, nrows=NUMBER_OF_ROWS)
# transform data
data = data.pivot(index=columns[0], columns=columns[1], values=columns[2])
# replace missing values with zeros
data = data.fillna(0)

# heat-map before DataFrame sort
sns.heatmap(data, yticklabels=False, xticklabels=False)
plt.savefig('./images/heatmap-before-sort.png')
plt.clf()

# Sort user-item DataFrame so that the density decreases along the main diagonal
matrix_data = data.as_matrix()
# Number of votes for each user
for i in range(0, matrix_data.shape[0]-1):
    row = matrix_data[i]
    no_of_votes = sum(k > 0 for k in row)
    print "The user with id: ", i+1, " voted ", no_of_votes, " times."
# Create matrix with UserID and number of votes
    Num_users, Num_votes = 84, 2976
    temp_matrix = np.ones((Num_users, Num_votes))
    temp_matrix = ([[i+1], [no_of_votes]])
    print temp_matrix
# TODO Sort matrix

#TODO Back to data frame

# heat-map after DataFrame sort
sns.heatmap(data, yticklabels=False, xticklabels=False)
plt.savefig('./images/heatmap-after-sort.png')

# data info
print 'Number of users: ', len(data.index)
print 'Number of items:', len(data.columns), '\n'

# memory-based algorithms

# model-based algorithms
kmeans_cluster = model_based.kmeans_clustering(data)
print 'K-means labels: ', kmeans_cluster.labels_, '\n'

agglomerative_cluster = model_based.agglomerative_clustering(data)
print 'Agglomerative labels: ', agglomerative_cluster.labels_, '\n'
