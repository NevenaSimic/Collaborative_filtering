# author: Zoran Luledzija
import math
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


def kmeans_clustering(data):
    """
    K-means clustering
    :param data:
    :return cluster:
    """
    # TODO 1a: Find optimal number of clusters
    k = int(math.ceil(len(data.index)*0.1))
    cluster = KMeans(n_clusters=k)
    cluster.fit(data)

    return cluster


def agglomerative_clustering(data):
    """
    Hierarchical agglomerative clustering
    :param data:
    :return cluster:
    """
    # TODO 1b: Find optimal number of clusters
    k = int(math.ceil(len(data.index)*0.1))
    cluster = AgglomerativeClustering(n_clusters=k)
    cluster.fit(data)

    return cluster
