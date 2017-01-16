import pandas as pd
import numpy as np
from sort import sort
from helpers import *
import model_based
import memory_based
import operator

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
print 'Number of users: ', len(data.index)
print 'Number of items: ', len(data.columns), '\n'

# item-based matrix
item_based_data = data.copy().T
save_heatmap_image(item_based_data, 'item-based-heatmap-before-sort')
item_based_data = sort(item_based_data)
save_heatmap_image(item_based_data, 'item-based-heatmap-after-sort')

# user-based matrix
user_based_data = data.copy()
save_heatmap_image(user_based_data, 'user-based-heatmap-before-sort')
user_based_data = sort(user_based_data)
save_heatmap_image(user_based_data, 'user-based-heatmap-after-sort')

###########################
# Memory-based algorithms #
###########################
matrix_data = data.as_matrix()
num_users = matrix_data.shape[0]
for user1 in range(num_users):
	for user2 in range(num_users):
		if user1 is not user2:
			rating_user1 = np.squeeze(np.asarray(matrix_data[user1]))
			rating_user2 = np.squeeze(np.asarray(matrix_data[user2]))
			euclidean_distance = memory_based.euclidean(rating_user1, rating_user2)
			#print 'Euclidean distance between users' ,user1, 'and', user2, 'is: ', euclidean_distance, '\n'


##########################
# Model-based algorithms #
##########################

# user-based approach
kmeans_cluster = model_based.kmeans_clustering(user_based_data)
print 'K-means labels: ', kmeans_cluster.labels_, '\n'
cluster_suggestions = {}
for label in kmeans_cluster.labels_:
    if label not in cluster_suggestions:
        users = user_based_data.iloc[kmeans_cluster.labels_ == label]
        suggestions = suggest(users)
        cluster_suggestions[label] = users.columns.values[max(suggestions.iteritems(), key=operator.itemgetter(1))[0]]
print 'Cluster suggestion: ', cluster_suggestions, '\n'

agglomerative_cluster = model_based.agglomerative_clustering(user_based_data)
print 'Agglomerative labels: ', agglomerative_cluster.labels_, '\n'
cluster_suggestions = {}
for label in agglomerative_cluster.labels_:
    if label not in cluster_suggestions:
        users = user_based_data.iloc[kmeans_cluster.labels_ == label]
        suggestions = suggest(users)
        cluster_suggestions[label] = users.columns.values[max(suggestions.iteritems(), key=operator.itemgetter(1))[0]]
print 'Cluster suggestion: ', cluster_suggestions, '\n'

# item-base approach
kmeans_cluster = model_based.kmeans_clustering(item_based_data)
print 'K-means labels: ', kmeans_cluster.labels_, '\n'
cluster_suggestions = {}
for label in kmeans_cluster.labels_:
    if label not in cluster_suggestions:
        items = item_based_data.iloc[kmeans_cluster.labels_ == label]
        suggestions = suggest_max_voted_item(items)
        cluster_suggestions[label] = items.index.values[max(suggestions.iteritems(), key=operator.itemgetter(1))[0]]
print 'Cluster suggestion: ', cluster_suggestions, '\n'

agglomerative_cluster = model_based.agglomerative_clustering(item_based_data)
print 'Agglomerative labels: ', agglomerative_cluster.labels_, '\n'
cluster_suggestions = {}
for label in agglomerative_cluster.labels_:
    if label not in cluster_suggestions:
        items = item_based_data.iloc[kmeans_cluster.labels_ == label]
        suggestions = suggest_max_voted_item(items)
        cluster_suggestions[label] = items.index.values[max(suggestions.iteritems(), key=operator.itemgetter(1))[0]]
print 'Cluster suggestion: ', cluster_suggestions, '\n'
