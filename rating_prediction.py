def average_rating_prediction(users, movie_id):
    rate = 0.0
    divisor = 0
    for i in range(users.shape[0]):
        if users.iloc[i][movie_id] > 0:
            rate += users.iloc[i][movie_id]
            divisor += 1
    if divisor > 0:
        rate /= divisor

    return rate


def similarity_rating_prediction(data, distances, column_id):
    rate = 0.0
    divisor = 0.0
    similarities = convert_distances_to_similarities(distances)

    for i in range(data.shape[0]):
        if data.iloc[i][column_id] > 0:
            rate += similarities[i] * data.iloc[i][column_id]
            divisor += similarities[i]
    if divisor > 0:
        rate /= divisor

    return rate


def adjusted_similarity_rating_prediction(user, neighbours, distances, movie_id):
    rate = 0.0
    divisor = 0.0
    average_rate = average_user_rate(user)
    similarities = convert_distances_to_similarities(distances)

    for i in range(neighbours.shape[0]):
        if neighbours.iloc[i][movie_id] > 0:
            rate += similarities[i] * (neighbours.iloc[i][movie_id] - average_user_rate(neighbours.iloc[i]))
            divisor += similarities[i]
    if divisor > 0:
        rate /= divisor
        rate += average_rate

    return rate


def average_user_rate(user):
    rating = 0.0
    divisor = 0
    for movie_rate in user:
        if movie_rate > 0:
            rating += movie_rate
            divisor += 1
    if divisor > 0:
        rating /= divisor

    return rating


def convert_distances_to_similarities(distances):
    dist = normalize_distances(distances)
    similarities = [(1 - d) for d in dist]

    return similarities


def normalize_distances(distances):
    normalized_distances = []
    min_value = min(distances)
    max_value = max(distances)

    if (max_value > 1) or (min_value < 0):
        for d in distances:
            similarity = (d - min_value) / (max_value - min_value)
            normalized_distances.append(similarity)
    else:
        normalized_distances = distances

    return normalized_distances
