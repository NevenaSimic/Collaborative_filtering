from math import sqrt
from time import time
import pandas as pd
import numpy as np
from sort import sort
from helpers import *
import model_based
import memory_based
from operator import itemgetter
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error

# number of rows from ratings.dat file which will be processed
NUMBER_OF_ROWS = 100000

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
print '*******************************************'
print '***  MEMORY-BASED(User-based approach)  ***'
print '*******************************************', '\n'

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
print '*******************************************'
print '***             MODEL-BASED             ***'
print '*******************************************', '\n'

# predict ratings for the given user and movies
user_id = 34
movies = [593, 110, 50, 457]
true_ratings = user_based_data.loc[34, movies].values.tolist()

# base kmeans cluster recommendation
start = time()
predicted_ratings = model_based.base_kmeans_cluster_recommendation(user_based_data, user_id, movies)
execution = time() - start
mae = mean_absolute_error(true_ratings, predicted_ratings)
rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))
print 'Base KMeans cluster recommendation:'
print 'True ratings: {}\nPredicted ratings: {}\nMAE: {}\nRMSE: {}\nExecution: {}\n'.format(true_ratings, predicted_ratings, mae, rmse, execution)

# knn kmeans cluster recommendation
start = time()
predicted_ratings = model_based.kmeans_cluster_recommendation(user_based_data, user_id, movies, 6)
execution = time() - start
mae = mean_absolute_error(true_ratings, predicted_ratings)
rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))
print 'KNN KMeans cluster recommendation:'
print 'True ratings: {}\nPredicted ratings: {}\nMAE: {}\nRMSE: {}\nExecution: {}\n'.format(true_ratings, predicted_ratings, mae, rmse, execution)

# base agglomerative cluster recommendation
start = time()
predicted_ratings = model_based.base_agglomerative_cluster_recommendation(user_based_data, user_id, movies)
execution = time() - start
mae = mean_absolute_error(true_ratings, predicted_ratings)
rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))
print 'Base Agglomerative cluster recommendation:'
print 'True ratings: {}\nPredicted ratings: {}\nMAE: {}\nRMSE: {}\nExecution: {}\n'.format(true_ratings, predicted_ratings, mae, rmse, execution)

# knn agglomerative cluster recommendation
start = time()
predicted_ratings = model_based.agglomerative_cluster_recommendation(user_based_data, user_id, movies, 6)
execution = time() - start
mae = mean_absolute_error(true_ratings, predicted_ratings)
rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))
print 'KNN Agglomerative cluster recommendation:'
print 'True ratings: {}\nPredicted ratings: {}\nMAE: {}\nRMSE: {}\nExecution: {}\n'.format(true_ratings, predicted_ratings, mae, rmse, execution)
