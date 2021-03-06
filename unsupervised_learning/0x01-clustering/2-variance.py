#!/usr/bin/env python3
"""
Clustering Module
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set:

    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the centroid means for
    each cluster
    You are not allowed to use any loops

    Returns: var, or None on failure
        var is the total variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    if C.shape[0] > X.shape[0]:
        return None

    dtcs = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
    mind = np.min(dtcs, axis=0)
    return np.sum(mind ** 2)
