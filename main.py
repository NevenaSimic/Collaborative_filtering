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

# sort
# TODO: Sort user-item DataFrame so that the density decreases along the main diagonal

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

