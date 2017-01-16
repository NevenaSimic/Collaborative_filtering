import pandas as pd
import numpy as np
from sort import sort
from helpers import *
import model_based
import memory_based
from operator import itemgetter
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

# number of rows from ratings.dat file which will be processed
NUMBER_OF_ROWS = 10000

columns = ['UserID', 'MovieID', 'Rating']
# read data
data = pd.read_csv('./data/ml-10M100K/ratings.dat', sep='::', engine='python', header=None, names=columns, usecols=columns, nrows=NUMBER_OF_ROWS)
# transform data
data = data.pivot(index=columns[0], columns=columns[1], values=columns[2])
# replace missing values with zeros
data = data.fillna(0)
# data info
print '\n', 'Number of users: ', len(data.index)
print 'Number of items: ', len(data.columns), '\n'

# Item-based matrix
item_based_data = data.copy().T
save_heatmap_image(item_based_data, 'item-based-heatmap-before-sort')
item_based_data = sort(item_based_data)
save_heatmap_image(item_based_data, 'item-based-heatmap-after-sort')

# User-based matrix
user_based_data = data.copy()
save_heatmap_image(user_based_data, 'user-based-heatmap-before-sort')
user_based_data = sort(user_based_data)
save_heatmap_image(user_based_data, 'user-based-heatmap-after-sort')

###########################
# Memory-based algorithms #
###########################

# User-based approach
print '\n', '********************************', '\n'
print 'MEMORY-BASED(User-based approach)', '\n'

nbrs_euclidean = NearestNeighbors(algorithm='brute', metric='euclidean')
nbrs_euclidean.fit(user_based_data)
query_index = np.random.choice(user_based_data.shape[0])
distances, indices = nbrs_euclidean.kneighbors(user_based_data.iloc[query_index, :].reshape(1, -1), n_neighbors=6)
for i in range(0, len(distances.flatten())):
    if i == 0:
        print 'Recommendations(EUCLIDEAN DISTANCE) for {0}:\n'.format(user_based_data.index[query_index])
    else:
        print '{1}, with distance of {2}:'.format(i, user_based_data.index[indices.flatten()[i]],distances.flatten()[i])

print '\n'

nbrs_cosine = NearestNeighbors(algorithm='brute', metric = 'cosine')
nbrs_cosine.fit(user_based_data)
query_index = np.random.choice(user_based_data.shape[0])
distances, indices = nbrs_cosine.kneighbors(user_based_data.iloc[query_index, :].reshape(1, -1), n_neighbors = 6)
for i in range(0, len(distances.flatten())):
    if i == 0:
        print 'Recommendations(COSINE SIMILARITY) for {0}:\n'.format(user_based_data.index[query_index])
    else:
        print '{1}, with distance of {2}:'.format(i, user_based_data.index[indices.flatten()[i]],distances.flatten()[i])

print '\n'

nbrs_jaccard = NearestNeighbors(algorithm='brute', metric='jaccard')
nbrs_jaccard.fit(user_based_data)
query_index = np.random.choice(user_based_data.shape[0])
distances, indices = nbrs_jaccard.kneighbors(user_based_data.iloc[query_index, :].reshape(1, -1), n_neighbors=6)
for i in range(0, len(distances.flatten())):
    if i == 0:
        print 'Recommendations(JACCARD SIMILARITY) for {0}:\n'.format(user_based_data.index[query_index])
    else:
        print '{1}, with distance of {2}:'.format(i, user_based_data.index[indices.flatten()[i]],distances.flatten()[i])



##########################
# Model-based algorithms #
##########################

# User-based approach
print '\n', '********************************', '\n'
print 'MODEL-BASED(User-based approach)', '\n'

kmeans_cluster = model_based.kmeans_clustering(user_based_data)
print 'K-means labels: ', kmeans_cluster.labels_, '\n'
cluster_suggestions = {}
for label in kmeans_cluster.labels_:
    if label not in cluster_suggestions:
        users = user_based_data.iloc[kmeans_cluster.labels_ == label]
        suggestions = suggest(users)
        cluster_suggestions[label] = users.columns.values[max(suggestions.iteritems(), key=itemgetter(1))[0]]
print 'Cluster suggestion: ', cluster_suggestions, '\n'

agglomerative_cluster = model_based.agglomerative_clustering(user_based_data)
print 'Agglomerative labels: ', agglomerative_cluster.labels_, '\n'
cluster_suggestions = {}
for label in agglomerative_cluster.labels_:
    if label not in cluster_suggestions:
        users = user_based_data.iloc[kmeans_cluster.labels_ == label]
        suggestions = suggest(users)
        cluster_suggestions[label] = users.columns.values[max(suggestions.iteritems(), key=itemgetter(1))[0]]
print 'Cluster suggestion: ', cluster_suggestions, '\n'

# Item-based approach
print '********************************', '\n'
print 'MODEL-BASED(Item-based approach)', '\n'

kmeans_cluster = model_based.kmeans_clustering(item_based_data)
print 'K-means labels: ', kmeans_cluster.labels_, '\n'
cluster_suggestions = {}
for label in kmeans_cluster.labels_:
    if label not in cluster_suggestions:
        items = item_based_data.iloc[kmeans_cluster.labels_ == label]
        suggestions = suggest_max_voted_item(items)
        cluster_suggestions[label] = items.index.values[max(suggestions.iteritems(), key=itemgetter(1))[0]]
print 'Cluster suggestion: ', cluster_suggestions, '\n'

agglomerative_cluster = model_based.agglomerative_clustering(item_based_data)
print 'Agglomerative labels: ', agglomerative_cluster.labels_, '\n'
cluster_suggestions = {}
for label in agglomerative_cluster.labels_:
    if label not in cluster_suggestions:
        items = item_based_data.iloc[kmeans_cluster.labels_ == label]
        suggestions = suggest_max_voted_item(items)
        cluster_suggestions[label] = items.index.values[max(suggestions.iteritems(), key=itemgetter(1))[0]]
print 'Cluster suggestion: ', cluster_suggestions, '\n'
