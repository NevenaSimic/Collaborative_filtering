# author: Nevena Simic
import numpy as np


def sort(data):
    n, m = data.shape

    # sort by rows
    matrix_data = data.as_matrix()
    matrix = np.zeros((n, 2), dtype=np.int32)
    for i in range(n):
        matrix[i][0] = data.index.values[i]
        matrix[i][1] = sum(k > 0 for k in matrix_data[i])

    matrix = matrix[matrix[:, 1].argsort()[::-1]]
    data = data.reindex(np.asarray(matrix[:, 0], dtype=np.int32))

    # sort by cols
    matrix_data = data.as_matrix()
    matrix = np.zeros((m, 2), dtype=np.int32)
    for i in range(m):
        matrix[i][0] = data.columns.values[i]
        matrix[i][1] = sum(k > 0 for k in matrix_data[:, i])

    matrix = matrix[matrix[:, 1].argsort()[::-1]]
    data = data.reindex(columns=np.asarray(matrix[:, 0], dtype=np.int32))

    return data
