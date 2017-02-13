from math import sqrt
from random import choice
from time import time

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import user_based
import item_based
import model_based
from helpers import *
from sort import sort

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
rows, cols = data.shape
print '\n', 'Number of users: ', rows
print 'Number of items: ', cols, '\n'

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

# predicting ratings
number_of_predictions = 10
predicted_ratings = {
    'user-based-euclidean': [],
    'user-based-jaccard': [],
    'user-based-cosine': [],
    'user-based-correlation': [],
    'item-based-euclidean': [],
    'item-based-jaccard': [],
    'item-based-cosine': [],
    'item-based-correlation': [],
    'base-kmeans-cluster': [],
    'knn-kmeans-cluster': [],
    'base-agglomerative-cluster': [],
    'knn-agglomerative-cluster': []
}
executions = {
    'user-based-euclidean': [],
    'user-based-jaccard': [],
    'user-based-cosine': [],
    'user-based-correlation': [],
    'item-based-euclidean': [],
    'item-based-jaccard': [],
    'item-based-cosine': [],
    'item-based-correlation': [],
    'base-kmeans-cluster': [],
    'knn-kmeans-cluster': [],
    'base-agglomerative-cluster': [],
    'knn-agglomerative-cluster': []
}
users = []
movies = []
true_ratings = []
for n in range(number_of_predictions):
    user_id = choice(list(data.index))
    movie_id = None
    for ind, val in data.loc[user_id, :].iteritems():
        if val > 0:
            movie_id = ind
            break
    if movie_id:
        users.append(user_id)
        movies.append(movie_id)
        true_ratings.append(data.loc[user_id, movie_id])

        # TODO: remove original rating

        start = time()
        predicted_rating = user_based.euclidean_knn_recommendation(user_based_data, user_id, movie_id, n=20)
        execution = time() - start
        predicted_ratings['user-based-euclidean'].append(predicted_rating)
        executions['user-based-euclidean'].append(execution)

        start = time()
        predicted_rating = user_based.jaccard_knn_recommendation(user_based_data, user_id, movie_id, n=20)
        execution = time() - start
        predicted_ratings['user-based-jaccard'].append(predicted_rating)
        executions['user-based-jaccard'].append(execution)

        start = time()
        predicted_rating = user_based.cosine_knn_recommendation(user_based_data, user_id, movie_id, n=20)
        execution = time() - start
        predicted_ratings['user-based-cosine'].append(predicted_rating)
        executions['user-based-cosine'].append(execution)

        start = time()
        predicted_rating = user_based.correlation_knn_recommendation(user_based_data, user_id, movie_id, n=20)
        execution = time() - start
        predicted_ratings['user-based-correlation'].append(predicted_rating)
        executions['user-based-correlation'].append(execution)

        start = time()
        predicted_rating = item_based.euclidean_knn_recommendation(item_based_data, user_id, movie_id, n=20)
        execution = time() - start
        predicted_ratings['item-based-euclidean'].append(predicted_rating)
        executions['item-based-euclidean'].append(execution)

        start = time()
        predicted_rating = item_based.jaccard_knn_recommendation(item_based_data, user_id, movie_id, n=20)
        execution = time() - start
        predicted_ratings['item-based-jaccard'].append(predicted_rating)
        executions['item-based-jaccard'].append(execution)

        start = time()
        predicted_rating = item_based.cosine_knn_recommendation(item_based_data, user_id, movie_id, n=20)
        execution = time() - start
        predicted_ratings['item-based-cosine'].append(predicted_rating)
        executions['item-based-cosine'].append(execution)

        start = time()
        predicted_rating = item_based.correlation_knn_recommendation(item_based_data, user_id, movie_id, n=20)
        execution = time() - start
        predicted_ratings['item-based-correlation'].append(predicted_rating)
        executions['item-based-correlation'].append(execution)

        start = time()
        predicted_rating = model_based.base_kmeans_cluster_recommendation(user_based_data, user_id, movie_id)
        execution = time() - start
        predicted_ratings['base-kmeans-cluster'].append(predicted_rating)
        executions['base-kmeans-cluster'].append(execution)

        start = time()
        predicted_rating = model_based.kmeans_cluster_recommendation(user_based_data, user_id, movie_id, n=20)
        execution = time() - start
        predicted_ratings['knn-kmeans-cluster'].append(predicted_rating)
        executions['knn-kmeans-cluster'].append(execution)

        start = time()
        predicted_rating = model_based.base_agglomerative_cluster_recommendation(user_based_data, user_id, movie_id)
        execution = time() - start
        predicted_ratings['base-agglomerative-cluster'].append(predicted_rating)
        executions['base-agglomerative-cluster'].append(execution)

        start = time()
        predicted_rating = model_based.agglomerative_cluster_recommendation(user_based_data, user_id, movie_id, n=20)
        execution = time() - start
        predicted_ratings['knn-agglomerative-cluster'].append(predicted_rating)
        executions['knn-agglomerative-cluster'].append(execution)


# display results:
for key, value in predicted_ratings.iteritems():
    mae = mean_absolute_error(true_ratings, value)
    rmse = sqrt(mean_squared_error(true_ratings, value))
    print key.upper()
    print 'True ratings: {}\nPredicted ratings: {}\nMAE: {}\nRMSE: {}\n'.format(true_ratings, value, mae, rmse)

for key, value in executions.iteritems():
    print key.upper()
    average_execution = sum(value) / len(value)
    print 'Average execution (s): {}\n'.format(average_execution)
