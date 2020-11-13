#!/usr/bin/env python3
"""
Clustering Module
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
    iterations that should be performed
    If no change in the cluster centroids occurs between iterations, your
    function should return
    Initialize the cluster centroids using a multivariate uniform distribution
    (based on0-initialize.py)
    If a cluster contains no data points during the update step, reinitialize
    its centroid
    You should use numpy.random.uniform exactly twice
    You may use at most 2 loops
    Returns: C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of the
        cluster in C that each data point belongs to

    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None

    n, d = X.shape
    xmax = np.max(X, axis=0).astype(np.float)
    xmin = np.min(X, axis=0).astype(np.float)
    C = np.random.uniform(xmin, xmax, size=(k, d))

    for i in range(iterations):
        dtcs = np.linalg.norm(X[:, None] - C, axis=-1)
        klss = np.argmin(dtcs, axis=-1)

        Cc = np.copy(C)
        for j in range(k):
            idxs = np.argwhere(klss == j)
            if not len(idxs):
                C[j] = np.random.uniform(xmin, xmax, size=(1, d))
            else:
                C[j] = np.mean(X[idxs], axis=0)
        if (Cc == C).all():
            return (C, klss)

    dtcs = np.linalg.norm(X[:, None] - C, axis=-1)
    klss = np.argmin(dtcs, axis=-1)

    return (C, klss)
