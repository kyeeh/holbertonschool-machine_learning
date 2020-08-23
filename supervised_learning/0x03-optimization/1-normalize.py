#!/usr/bin/env python3
"""
Optimization Module
"""
import numpy as np


def normalize(X, m, s):
    """
    X is the numpy.ndarray of shape (d, nx) to normalize

        d is the number of data points
        nx is the number of features

    m is a np.ndarray of shape (nx,) that contains the mean of features of X
    s is a np.ndarray of shape (nx,) that contains the stdv of features of X

    Returns: The normalized X matrix
    """
    return (X - m) / s
