#!/usr/bin/env python3
"""
Clustering Module
"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM:

    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster
    You may use at most 1 loop

    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the updated priors for
        each cluster
        m is a numpy.ndarray of shape (k, d) containing the updated centroid
        means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the updated
        covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    n, d = X.shape
    k, _ = g.shape

    if not np.isclose(np.sum(g, axis=0), np.ones((n, ))).all():
        return None, None, None
    pi, m, s = np.zeros((k,)), np.zeros((k, d)), np.zeros((k, d, d))

    for i in range(k):
        m[i] = np.dot(g[i], X) / np.sum(g[i])
        xmm = X - m[i]
        s[i] = np.dot(g[i] * xmm.T, xmm) / np.sum(g[i])
        pi[i] = np.sum(g[i]) / n
    return pi, m, s
