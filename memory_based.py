# author: Nevena Simic
from rating_predictions import similarity_rating_prediction
from sklearn.neighbors import NearestNeighbors


def euclidean_knn_recommendation(data, user_id, movies, n=5):
    nn = NearestNeighbors(algorithm='brute', metric='euclidean', n_neighbors=n)
    nn.fit(data)
    distances, indices = nn.kneighbors(data.ix[user_id, :].values.reshape(1, -1))
    distances = distances.tolist()[0]
    indices = indices.tolist()[0]

    ratings = []
    for movie_id in movies:
        ratings.append(similarity_rating_prediction(data.iloc[indices, :], distances, movie_id))

    return ratings


def jaccard_knn_recommendation(data, user_id, movies, n=5):
    nn = NearestNeighbors(algorithm='brute', metric='jaccard', n_neighbors=n)
    nn.fit(data)
    distances, indices = nn.kneighbors(data.ix[user_id, :].values.reshape(1, -1))
    distances = distances.tolist()[0]
    indices = indices.tolist()[0]

    ratings = []
    for movie_id in movies:
        ratings.append(similarity_rating_prediction(data.iloc[indices, :], distances, movie_id))

    return ratings


def cosine_knn_recommendation(data, user_id, movies, n=5):
    nn = NearestNeighbors(algorithm='brute', metric='cosine', n_neighbors=n)
    nn.fit(data)
    distances, indices = nn.kneighbors(data.ix[user_id, :].values.reshape(1, -1))
    distances = distances.tolist()[0]
    indices = indices.tolist()[0]

    ratings = []
    for movie_id in movies:
        ratings.append(similarity_rating_prediction(data.iloc[indices, :], distances, movie_id))

    return ratings


def pearson_knn_recommendation(data, user_id, movies, n=5):
    nn = NearestNeighbors(algorithm='brute', metric='correlation', n_neighbors=n)
    nn.fit(data)
    distances, indices = nn.kneighbors(data.ix[user_id, :].values.reshape(1, -1))
    distances = distances.tolist()[0]
    indices = indices.tolist()[0]

    ratings = []
    for movie_id in movies:
        ratings.append(similarity_rating_prediction(data.iloc[indices, :], distances, movie_id))

    return ratings
