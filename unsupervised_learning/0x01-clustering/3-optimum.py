#!/usr/bin/env python3
"""
Clustering Module
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance:

    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of clusters to
    check for (inclusive)
    kmax is a positive integer containing the maximum number of clusters to
    check for (inclusive)
    iterations is a positive integer containing the maximum number of
    iterations for K-means
    This function should analyze at least 2 different cluster sizes
    You may use at most 2 loops

    Returns: results, d_vars, or None, None on failure
        results is a list containing the outputs of K-means for each cluster
        size
        d_vars is a list containing the difference in variance from the
        smallest cluster size for each cluster size
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None)
    if type(kmin) is not int:
        return (None, None)
    if type(iterations) is not int:
        return (None, None)
    if kmax is not None and type(kmax) is not int:
        return (None, None)
    n, _ = X.shape
    if kmax is None:
        kmax = n
    if kmin <= 0 or kmax <= 0 or iterations <= 0:
        return (None, None)
    if kmin >= kmax:
        return (None, None)

    d_vars = []
    results = []
    for i in range(kmin, kmax + 1):
        center, klss = kmeans(X, i, iterations)
        results.append((center, klss))
        if i == kmin:
            kmin_var = variance(X, center)
        cvar = variance(X, center)
        d_vars.append(kmin_var - cvar)
    return (results, d_vars)
