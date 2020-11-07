#!/usr/bin/env python3
"""
Clustering Module
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution:

    X is a numpy.ndarray of shape (n, d) containing the data points whose PDF
    should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance of the
    distribution
    You are not allowed to use any loops
    You are not allowed to use the function numpy.diag or the method
    numpy.ndarray.diagonal

    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values for each
        data point
    All values in P should have a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1] or X.shape[1] != S.shape[1]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    n, d = X.shape
    Xmm = X - m
    Sinv = np.linalg.inv(S)

    p = 1. / (np.sqrt(((2 * np.pi)**d * np.linalg.det(S))))
    fac = np.einsum('...k,kl,...l->...', Xmm, Sinv, Xmm)
    q = np.exp(-fac / 2)
    return (np.maximum(p*q, 1e-300))
