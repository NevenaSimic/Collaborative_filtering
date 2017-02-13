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


def similarity_rating_prediction(data, similarities, column_id):
    rate = 0.0
    divisor = 0.0
    for i in range(data.shape[0]):
        if data.iloc[i][column_id] > 0:
            rate += similarities[i] * data.iloc[i][column_id]
            divisor += similarities[i]
    if divisor > 0:
        rate /= divisor

    return rate


def adjusted_similarity_rating_prediction(user, neighbours, similarities, movie_id):
    rate = 0.0
    divisor = 0.0
    average_rate = average_user_rate(user)

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
