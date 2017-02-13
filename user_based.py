from sklearn.neighbors import NearestNeighbors

from rating_prediction import adjusted_similarity_rating_prediction


def euclidean_knn_recommendation(data, user_id, movie_id, n=5):
    nn = NearestNeighbors(algorithm='brute', metric='euclidean', n_neighbors=n)
    nn.fit(data.drop(user_id))
    distances, indices = nn.kneighbors(data.ix[user_id, :].values.reshape(1, -1))
    distances = distances.tolist()[0]
    indices = indices.tolist()[0]

    return adjusted_similarity_rating_prediction(data.loc[user_id], data.iloc[indices, :], distances, movie_id)


def jaccard_knn_recommendation(data, user_id, movie_id, n=5):
    nn = NearestNeighbors(algorithm='brute', metric='jaccard', n_neighbors=n)
    nn.fit(data.drop(user_id))
    distances, indices = nn.kneighbors(data.ix[user_id, :].values.reshape(1, -1))
    distances = distances.tolist()[0]
    indices = indices.tolist()[0]

    return adjusted_similarity_rating_prediction(data.loc[user_id], data.iloc[indices, :], distances, movie_id)


def cosine_knn_recommendation(data, user_id, movie_id, n=5):
    nn = NearestNeighbors(algorithm='brute', metric='cosine', n_neighbors=n)
    nn.fit(data.drop(user_id))
    distances, indices = nn.kneighbors(data.ix[user_id, :].values.reshape(1, -1))
    distances = distances.tolist()[0]
    indices = indices.tolist()[0]

    return adjusted_similarity_rating_prediction(data.loc[user_id], data.iloc[indices, :], distances, movie_id)


def correlation_knn_recommendation(data, user_id, movie_id, n=5):
    nn = NearestNeighbors(algorithm='brute', metric='correlation', n_neighbors=n)
    nn.fit(data.drop(user_id))
    distances, indices = nn.kneighbors(data.ix[user_id, :].values.reshape(1, -1))
    distances = [(1 - x) for x in distances.tolist()[0]]
    indices = indices.tolist()[0]

    return adjusted_similarity_rating_prediction(data.loc[user_id], data.iloc[indices, :], distances, movie_id)
