#!/usr/bin/env python3
"""
Clustering Module
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model:

    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    You are not allowed to use any loops

    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the priors for each
        cluster, initialized evenly
        m is a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster, initialized with K-means
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for each cluster, initialized as identity matrices
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None, None
        if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
            return None, None, None
        pi = np.full((k, ), 1 / k)
        m, _ = kmeans(X, k)
        d = X.shape[1]
        S = np.full((k, d, d), np.identity(d))
        return pi, m, S
    except Exception:
        return None, None, None
