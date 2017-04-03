import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from random import choice


def save_heatmap_image(data, file_name):
    sns.heatmap(data, yticklabels=False, xticklabels=False)
    plt.savefig('./images/' + file_name + '.png')
    plt.clf()


def generate_rating_density_image(data):
    n, m = data.shape
    d = 5
    row_step = n / d
    column_step = m / d

    non_cumulative_rating_density_matrix = np.zeros((d, d))
    cumulative_rating_density_matrix = np.zeros((d, d))

    # non-cumulative
    for i in range(d):
        for j in range(d):
            chunk = data.iloc[i*row_step:(i+1)*row_step, j*column_step:(j+1)*column_step]
            rows, cols = chunk.shape
            number_of_votes = (chunk > 0).as_matrix().sum()
            density = (float(number_of_votes) / (rows * cols)) * 100
            non_cumulative_rating_density_matrix[i, j] = density

    sns.heatmap(non_cumulative_rating_density_matrix,
                yticklabels=range(row_step, n + 1, row_step),
                xticklabels=range(column_step, m + 1, column_step),
                annot=True,
                cmap="BuGn",
                cbar=False,
                linewidths=.5)
    plt.savefig('./images/non-cumulative-rating-density.png')
    plt.clf()

    # cumulative
    for i in range(d):
        for j in range(d):
            matrix = non_cumulative_rating_density_matrix[0:i+1, 0:j+1]
            s = matrix.sum()
            div = matrix.shape[0] * matrix.shape[1]
            cumulative_rating_density_matrix[i, j] = s / div

    sns.heatmap(cumulative_rating_density_matrix,
                yticklabels=range(row_step, n + 1, row_step),
                xticklabels=range(column_step, m + 1, column_step),
                annot=True,
                cmap="BuGn",
                cbar=False,
                linewidths=.5)
    plt.savefig('./images/cumulative-rating-density.png')
    plt.clf()


def save_graphic_results(file, data, xlabel, ylabel, n=5):
    fig, ax = plt.subplots()
    ax.set_color_cycle([
        'red',
        'black',
        'yellow',
        'gray',
        'green',
        'pink',
        'darkorange',
        'lightblue',
        'darkblue',
        'magenta',
        'olive',
        'brown'
    ])

    for key, value in data.iteritems():
        plt.plot(range(1, n + 1), value, label=key)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(file)
    plt.clf()


def get_data_for_testing(data, number_of_predictions=10):
    analized_pairs = []
    true_ratings = []
    for n in range(number_of_predictions):
        user_id = choice(list(data.index))
        movie_id = None
        for ind, val in data.loc[user_id, :].iteritems():
            if (val > 0) and ((user_id, movie_id) not in analized_pairs):
                movie_id = ind
                break
        if movie_id:
            analized_pairs.append((user_id, movie_id))
            true_ratings.append(data.loc[user_id, movie_id])

    return [analized_pairs, true_ratings]
