from rating_predictions import *
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors


def base_kmeans_cluster_recommendation(data, user_id, movies):
    k = 10
    cluster = KMeans(n_clusters=k)
    cluster.fit(data)

    label = cluster.labels_[data.index.get_loc(user_id)]
    users = data.iloc[cluster.labels_ == label]
    users = users.drop(user_id)

    ratings = [0.0] * len(movies)
    if users.shape[0] > 0:
        for i in range(len(movies)):
            ratings[i] = average_rating_prediction(users, movies[i])

    return ratings


def base_agglomerative_cluster_recommendation(data, user_id, movies):
    k = 10
    cluster = AgglomerativeClustering(n_clusters=k)
    cluster.fit(data)

    label = cluster.labels_[data.index.get_loc(user_id)]
    users = data.iloc[cluster.labels_ == label]
    users = users.drop(user_id)

    ratings = [0.0] * len(movies)
    if users.shape[0] > 0:
        for i in range(len(movies)):
            ratings[i] = average_rating_prediction(users, movies[i])

    return ratings


def kmeans_cluster_recommendation(data, user_id, movies, n=5):
    k = 10
    cluster = KMeans(n_clusters=k)
    cluster.fit(data)

    label = cluster.labels_[data.index.get_loc(user_id)]
    users = data.iloc[cluster.labels_ == label]
    if users.shape[0] < n:
        n = users.shape[0]

    nn = NearestNeighbors(algorithm='brute', metric='cosine', n_neighbors=n)
    nn.fit(users)
    distances, indices = nn.kneighbors(data.ix[user_id, :].values.reshape(1, -1))
    distances = distances.tolist()[0]
    indices = indices.tolist()[0]

    ratings = []
    for movie_id in movies:
        ratings.append(similarity_rating_prediction(users.iloc[indices, :], distances, movie_id))

    return ratings


def agglomerative_cluster_recommendation(data, user_id, movies, n=5):
    k = 10
    cluster = AgglomerativeClustering(n_clusters=k)
    cluster.fit(data)

    label = cluster.labels_[data.index.get_loc(user_id)]
    users = data.iloc[cluster.labels_ == label]
    if users.shape[0] < n:
        n = users.shape[0]

    nn = NearestNeighbors(algorithm='brute', metric='cosine', n_neighbors=n)
    nn.fit(users)
    distances, indices = nn.kneighbors(data.ix[user_id, :].values.reshape(1, -1))
    distances = distances.tolist()[0]
    indices = indices.tolist()[0]

    ratings = []
    for movie_id in movies:
        ratings.append(similarity_rating_prediction(users.iloc[indices, :], distances, movie_id))

    return ratings
