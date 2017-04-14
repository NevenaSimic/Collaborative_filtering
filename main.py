from math import sqrt
from time import time
import copy

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

# user-based matrix
user_based_data = data.copy()
save_heatmap_image(user_based_data, 'user-based-heatmap-before-sort')
user_based_data = sort(user_based_data)
save_heatmap_image(user_based_data, 'user-based-heatmap-after-sort')

# generate rating density image
generate_rating_density_image(user_based_data)


###########################################
# test collaborative filtering algorithms #
###########################################
step = 5
row_step = rows / step
column_step = cols / step

# predicting ratings
number_of_predictions = 100
algorithms = {
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
results = {
    'user': {
        'results': [],
    },
    'item': {
        'results': [],
    },
    'both': {
        'results': [],
    }
}


def perform_analysis(matrix, n):
    user_based_matrix = matrix
    item_based_matrix = user_based_matrix.copy().T

    pairs, ratings = get_data_for_testing(user_based_matrix, number_of_predictions)
    predicted_ratings = copy.deepcopy(algorithms)
    executions = copy.deepcopy(algorithms)

    for pair in pairs:
        user_id = pair[0]
        movie_id = pair[1]

        start = time()
        predicted_rating = user_based.euclidean_knn_recommendation(user_based_matrix, user_id, movie_id, n=20)
        execution_time = time() - start
        predicted_ratings['user-based-euclidean'].append(predicted_rating)
        executions['user-based-euclidean'].append(execution_time)

        start = time()
        predicted_rating = user_based.jaccard_knn_recommendation(user_based_matrix, user_id, movie_id, n=20)
        execution_time = time() - start
        predicted_ratings['user-based-jaccard'].append(predicted_rating)
        executions['user-based-jaccard'].append(execution_time)

        start = time()
        predicted_rating = user_based.cosine_knn_recommendation(user_based_matrix, user_id, movie_id, n=20)
        execution_time = time() - start
        predicted_ratings['user-based-cosine'].append(predicted_rating)
        executions['user-based-cosine'].append(execution_time)

        start = time()
        predicted_rating = user_based.correlation_knn_recommendation(user_based_matrix, user_id, movie_id, n=20)
        execution_time = time() - start
        predicted_ratings['user-based-correlation'].append(predicted_rating)
        executions['user-based-correlation'].append(execution_time)

        start = time()
        predicted_rating = item_based.euclidean_knn_recommendation(item_based_matrix, user_id, movie_id, n=20)
        execution_time = time() - start
        predicted_ratings['item-based-euclidean'].append(predicted_rating)
        executions['item-based-euclidean'].append(execution_time)

        start = time()
        predicted_rating = item_based.jaccard_knn_recommendation(item_based_matrix, user_id, movie_id, n=20)
        execution_time = time() - start
        predicted_ratings['item-based-jaccard'].append(predicted_rating)
        executions['item-based-jaccard'].append(execution_time)

        start = time()
        predicted_rating = item_based.cosine_knn_recommendation(item_based_matrix, user_id, movie_id, n=20)
        execution_time = time() - start
        predicted_ratings['item-based-cosine'].append(predicted_rating)
        executions['item-based-cosine'].append(execution_time)

        start = time()
        predicted_rating = item_based.correlation_knn_recommendation(item_based_matrix, user_id, movie_id, n=20)
        execution_time = time() - start
        predicted_ratings['item-based-correlation'].append(predicted_rating)
        executions['item-based-correlation'].append(execution_time)

        start = time()
        predicted_rating = model_based.base_kmeans_cluster_recommendation(user_based_matrix, user_id, movie_id)
        execution_time = time() - start
        predicted_ratings['base-kmeans-cluster'].append(predicted_rating)
        executions['base-kmeans-cluster'].append(execution_time)

        start = time()
        predicted_rating = model_based.kmeans_cluster_recommendation(user_based_matrix, user_id, movie_id, n=20)
        execution_time = time() - start
        predicted_ratings['knn-kmeans-cluster'].append(predicted_rating)
        executions['knn-kmeans-cluster'].append(execution_time)

        start = time()
        predicted_rating = model_based.base_agglomerative_cluster_recommendation(user_based_matrix, user_id, movie_id)
        execution_time = time() - start
        predicted_ratings['base-agglomerative-cluster'].append(predicted_rating)
        executions['base-agglomerative-cluster'].append(execution_time)

        start = time()
        predicted_rating = model_based.agglomerative_cluster_recommendation(user_based_matrix, user_id, movie_id, n=20)
        execution_time = time() - start
        predicted_ratings['knn-agglomerative-cluster'].append(predicted_rating)
        executions['knn-agglomerative-cluster'].append(execution_time)

    return {
        'step': n,
        'true_ratings': ratings,
        'predicted_ratings': predicted_ratings,
        'executions': executions
    }


for i in range(step):
    # perform analysis related to user number and density
    data = user_based_data.iloc[0:row_step, 0:(i + 1) * column_step]
    results['user']['results'].append(perform_analysis(data, n=i))

    # perform analysis related to item number and density
    data = user_based_data.iloc[0:(i + 1) * row_step, 0:column_step]
    results['item']['results'].append(perform_analysis(data, n=i))

    # perform analysis related to user and item number and density
    data = user_based_data.iloc[0:(i + 1) * row_step, 0:(i + 1) * column_step]
    results['both']['results'].append(perform_analysis(data, n=i))

# convert generated results into images
for key, value in results.iteritems():
    mae = copy.deepcopy(algorithms)
    rmse = copy.deepcopy(algorithms)
    execution = copy.deepcopy(algorithms)

    for result in results[key]['results']:
        true_ratings = result['true_ratings']

        for k, val in result['predicted_ratings'].iteritems():
            mae[k].append(mean_absolute_error(true_ratings, val))
            rmse[k].append(sqrt(mean_squared_error(true_ratings, val)))
            execution[k].append(sum(val) / len(val))

    # save MAE results
    save_graphic_results('./images/' + key + '_mae.png', mae, 'Size', 'MAE', step)
    # save RMSE results
    save_graphic_results('./images/' + key + '_rmse.png', rmse, 'Size', 'RMSE', step)
    # save execution results
    save_graphic_results('./images/' + key + '_execution.png', execution, 'Size', 'Average execution time', step)
