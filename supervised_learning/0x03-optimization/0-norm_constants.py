#!/usr/bin/env python3
"""
Optimization Module
"""
import numpy as np


def normalization_constants(X):
    """
    X is the numpy.ndarray of shape (m, nx) to normalize
        m is the number of data points
        nx is the number of features

    Returns: the mean and standard deviation of each feature, respectively
    """
    mean = np.sum(X, axis=0) / X.shape[0]
    vrnc = np.sum((X - mean) ** 2, axis=0) / X.shape[0]
    stdv = np.sqrt(vrnc)
    return mean, stdv
