#!/usr/bin/env python3
"""
Clustering Module
"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters

    Returns: pi, m, S, clss, bic
        pi is a numpy.ndarray of shape (k,) containing the cluster priors
        m is a numpy.ndarray of shape (k, d) containing the centroid means
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices
        clss is a numpy.ndarray of shape (n,) containing the cluster indices
        for each data point
        bic is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
        value for each cluster size tested
    """
    gmix = sklearn.mixture.GaussianMixture(n_components=k)
    gmix_fit = gmix.fit(X)
    m = gmix_fit.means_
    S = gmix_fit.covariances_
    pi = gmix_fit.weights_
    klss = gmix.predict(X)
    bic = gmix.bic(X)
    return (pi, m, S, klss, bic)
