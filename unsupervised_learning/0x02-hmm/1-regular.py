#!/usr/bin/env python3
"""
Markov Module
"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain:

    P is a is a square 2D numpy.ndarray of shape (n, n) representing the
    transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain

    Returns: a numpy.ndarray of shape (1, n) containing the steady state
    probabilities, or None on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None

    s = np.ones((1, P.shape[0])) / P.shape[0]
    Ps = P.copy()
    while True:
        sprv = s
        s = np.matmul(s, P)
        Ps = P * Ps
        if np.any(Ps <= 0):
            return (None)
        if np.all(sprv == s):
            return (s)
