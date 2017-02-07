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


def similarity_rating_prediction(users, distances, movie_id):
    rate = 0.0
    divisor = 0.0
    for i in range(users.shape[0]):
        if users.iloc[i][movie_id] > 0:
            rate += distances[i] * users.iloc[i][movie_id]
            divisor += distances[i]
    if divisor > 0:
        rate /= divisor

    return rate
