from math import sqrt
from time import time
import pandas as pd
from sort import sort
from helpers import *
import model_based
import memory_based
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


# predict ratings for the given user and movies
user_id = 34
movies = [593, 110, 50, 457, 1, 150, 608, 377, 296, 356]
true_ratings = user_based_data.loc[user_id, movies].values.tolist()


#############################
#  Memory-based algorithms  #
#############################
# user-based
start = time()
predicted_ratings = memory_based.euclidean_knn_recommendation(user_based_data, user_id, movies, n=6)
execution = time() - start
mae = mean_absolute_error(true_ratings, predicted_ratings)
rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))
print 'Euclidean kNN recommendation:'
print 'True ratings: {}\nPredicted ratings: {}\nMAE: {}\nRMSE: {}\nExecution: {}\n'.format(true_ratings, predicted_ratings, mae, rmse, execution)

start = time()
predicted_ratings = memory_based.jaccard_knn_recommendation(user_based_data, user_id, movies, n=6)
execution = time() - start
mae = mean_absolute_error(true_ratings, predicted_ratings)
rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))
print 'Jaccard kNN recommendation:'
print 'True ratings: {}\nPredicted ratings: {}\nMAE: {}\nRMSE: {}\nExecution: {}\n'.format(true_ratings, predicted_ratings, mae, rmse, execution)

start = time()
predicted_ratings = memory_based.cosine_knn_recommendation(user_based_data, user_id, movies, n=6)
execution = time() - start
mae = mean_absolute_error(true_ratings, predicted_ratings)
rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))
print 'Cosine kNN recommendation:'
print 'True ratings: {}\nPredicted ratings: {}\nMAE: {}\nRMSE: {}\nExecution: {}\n'.format(true_ratings, predicted_ratings, mae, rmse, execution)

start = time()
predicted_ratings = memory_based.pearson_knn_recommendation(user_based_data, user_id, movies, n=6)
execution = time() - start
mae = mean_absolute_error(true_ratings, predicted_ratings)
rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))
print 'Pearson kNN recommendation:'
print 'True ratings: {}\nPredicted ratings: {}\nMAE: {}\nRMSE: {}\nExecution: {}\n'.format(true_ratings, predicted_ratings, mae, rmse, execution)

# item-based
# TODO: implement

#############################
#   Model-based algorithms  #
#############################
start = time()
predicted_ratings = model_based.base_kmeans_cluster_recommendation(user_based_data, user_id, movies)
execution = time() - start
mae = mean_absolute_error(true_ratings, predicted_ratings)
rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))
print 'Base KMeans cluster recommendation:'
print 'True ratings: {}\nPredicted ratings: {}\nMAE: {}\nRMSE: {}\nExecution: {}\n'.format(true_ratings, predicted_ratings, mae, rmse, execution)

start = time()
predicted_ratings = model_based.kmeans_cluster_recommendation(user_based_data, user_id, movies, 6)
execution = time() - start
mae = mean_absolute_error(true_ratings, predicted_ratings)
rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))
print 'KNN KMeans cluster recommendation:'
print 'True ratings: {}\nPredicted ratings: {}\nMAE: {}\nRMSE: {}\nExecution: {}\n'.format(true_ratings, predicted_ratings, mae, rmse, execution)

start = time()
predicted_ratings = model_based.base_agglomerative_cluster_recommendation(user_based_data, user_id, movies)
execution = time() - start
mae = mean_absolute_error(true_ratings, predicted_ratings)
rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))
print 'Base Agglomerative cluster recommendation:'
print 'True ratings: {}\nPredicted ratings: {}\nMAE: {}\nRMSE: {}\nExecution: {}\n'.format(true_ratings, predicted_ratings, mae, rmse, execution)

start = time()
predicted_ratings = model_based.agglomerative_cluster_recommendation(user_based_data, user_id, movies, 6)
execution = time() - start
mae = mean_absolute_error(true_ratings, predicted_ratings)
rmse = sqrt(mean_squared_error(true_ratings, predicted_ratings))
print 'KNN Agglomerative cluster recommendation:'
print 'True ratings: {}\nPredicted ratings: {}\nMAE: {}\nRMSE: {}\nExecution: {}\n'.format(true_ratings, predicted_ratings, mae, rmse, execution)
