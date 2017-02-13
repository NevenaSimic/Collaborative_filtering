from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from rating_prediction import average_rating_prediction, adjusted_similarity_rating_prediction


def base_kmeans_cluster_recommendation(data, user_id, movie_id):
    k = 10
    cluster = KMeans(n_clusters=k)
    cluster.fit(data)

    label = cluster.labels_[data.index.get_loc(user_id)]
    users = data.iloc[cluster.labels_ == label]
    users = users.drop(user_id)

    rating = 0
    if users.shape[0] > 0:
        rating = average_rating_prediction(users, movie_id)

    return rating


def base_agglomerative_cluster_recommendation(data, user_id, movie_id):
    k = 10
    cluster = AgglomerativeClustering(n_clusters=k)
    cluster.fit(data)

    label = cluster.labels_[data.index.get_loc(user_id)]
    users = data.iloc[cluster.labels_ == label]
    users = users.drop(user_id)

    rating = 0
    if users.shape[0] > 0:
        rating = average_rating_prediction(users, movie_id)

    return rating


def kmeans_cluster_recommendation(data, user_id, movie_id, n=5):
    k = 10
    cluster = KMeans(n_clusters=k)
    cluster.fit(data)

    label = cluster.labels_[data.index.get_loc(user_id)]
    users = data.iloc[cluster.labels_ == label]
    if (users.shape[0] - 1) < n:
        n = users.shape[0] - 1

    nn = NearestNeighbors(algorithm='brute', metric='cosine', n_neighbors=n)
    nn.fit(users.drop(user_id))
    distances, indices = nn.kneighbors(data.ix[user_id, :].values.reshape(1, -1))
    distances = distances.tolist()[0]
    indices = indices.tolist()[0]

    rating = 0
    if users.shape[0] > 0:
        rating = adjusted_similarity_rating_prediction(data.loc[user_id], users.iloc[indices, :], distances, movie_id)

    return rating


def agglomerative_cluster_recommendation(data, user_id, movie_id, n=5):
    k = 10
    cluster = AgglomerativeClustering(n_clusters=k)
    cluster.fit(data)

    label = cluster.labels_[data.index.get_loc(user_id)]
    users = data.iloc[cluster.labels_ == label]
    if (users.shape[0] - 1) < n:
        n = users.shape[0] - 1

    nn = NearestNeighbors(algorithm='brute', metric='cosine', n_neighbors=n)
    nn.fit(users.drop(user_id))
    distances, indices = nn.kneighbors(data.ix[user_id, :].values.reshape(1, -1))
    distances = distances.tolist()[0]
    indices = indices.tolist()[0]

    rating = 0
    if users.shape[0] > 0:
        rating = adjusted_similarity_rating_prediction(data.loc[user_id], users.iloc[indices, :], distances, movie_id)

    return rating
