# author: Nevena Simic
import numpy as np


def sort(data):
    matrix_data = data.as_matrix()
    # Number of votes for each user
    matrix = np.zeros((matrix_data.shape[0], 2))
    for i in range(0, matrix_data.shape[0]):
        row = matrix_data[i]
        no_of_votes = sum(k > 0 for k in row)
        # print "The user with id: ", data.index.values[i], " voted ", no_of_votes, " times."
        # Create matrix with UserID and number of votes
        matrix[i][0] = data.index.values[i]
        matrix[i][1] = no_of_votes

    # Sorted matrix with UserID and number of votes
    matrix = matrix[matrix[:, 1].argsort()[::-1]]
    matrix_column = np.asarray(matrix[:, 0], dtype=np.int32)
    # Sorted user-item matrix so that the density decreases along the main diagonal
    data = data.reindex(matrix_column)

    return data
