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

# Data frame to matrix
matrix_data = data.as_matrix()
# Number of votes for each user
matrix = np.zeros((matrix_data.shape[0], 2))
for i in range(0, matrix_data.shape[0]-1):
    row = matrix_data[i]
    no_of_votes = sum(k > 0 for k in row)
    #print "The user with id: ", i+1, " voted ", no_of_votes, " times."

# Create matrix with UserID and number of votes
    matrix[i][0] = i+1
    matrix[i][1] = no_of_votes
print "*********Original matrix********"
print matrix
print "********************************\n\n"
print "*********Sorted matrix**********"
matrix = matrix[matrix[:, 1].argsort()[::-1]]
print matrix
print "********************************\n\n"

#Back to data frame
matrix_column = matrix[:, 0]
data = pandas.DataFrame(matrix_data)
data = data.set_index(matrix_column)
print data

#TODO Sort values

# heat-map after DataFrame sort
sns.heatmap(data, yticklabels=False, xticklabels=False)
plt.savefig('./images/heatmap-after-sort.png')

# data info
print 'Number of users: ', len(data.index)
print 'Number of items:', len(data.columns), '\n'

# Memory-based algorithms
num_users = matrix_data.shape[0]
print "\n\n**********************************\n\n"
for user1 in range(num_users):
    for user2 in range(num_users):
        if user1 is not user2:
            rating_user1 = np.squeeze(np.asarray(matrix_data[user1]))
            rating_user2 = np.squeeze(np.asarray(matrix_data[user2]))
euclidean_distance = memory_based.euclidean(rating_user1, rating_user2)
print 'Euclidean distance: ' , euclidean_distance, '\n'

# Model-based algorithms
kmeans_cluster = model_based.kmeans_clustering(data)
print 'K-means labels: ', kmeans_cluster.labels_, '\n'

agglomerative_cluster = model_based.agglomerative_clustering(data)
print 'Agglomerative labels: ', agglomerative_cluster.labels_, '\n'
