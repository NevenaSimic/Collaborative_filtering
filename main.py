import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import model_based

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
# TODO sort


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

