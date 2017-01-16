import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def save_heatmap_image(data, file_name):
    sns.heatmap(data, yticklabels=False, xticklabels=False)
    plt.savefig('./images/' + file_name + '.png')
    plt.clf()


def suggest(data):
    matrix = data.as_matrix()
    (n, m) = matrix.shape
    indices = {}

    for i in range(n):
        _indices = np.where(matrix[i, :] == max(matrix[i, :]))[0]
        for j in _indices:
            if j in indices:
                indices[j] += 1
            else:
                indices[j] = 1

    return indices


def suggest_max_voted_item(data):
    matrix = data.as_matrix()
    (n, m) = matrix.shape

    votes = {}
    for i in range(n):
        _indices = np.where(matrix[i, :] == max(matrix[i, :]))[0]
        number_of_max_votes = len(_indices)
        votes[i] = number_of_max_votes

    return votes