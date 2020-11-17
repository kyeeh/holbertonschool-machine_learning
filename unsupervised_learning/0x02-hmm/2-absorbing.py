#!/usr/bin/env python3
"""
Markov Module
"""
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing:

    P is a is a square 2D numpy.ndarray of shape (n, n) representing the
    standard transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain

    Returns: True if it is absorbing, or False on failure
    """
    if (np.all(np.diag(P) == 1)):
        return True
    if not np.any(np.diagonal(P) == 1):
        return False

    if np.any(np.diag(P) == 1):
        for i, row in enumerate(P):
            for j, col in enumerate(row):
                if i == j and ((i + 1) < len(P)) and ((j + 1) < len(P)):
                    if P[i + 1][j] == 0 and P[i][j + 1] == 0:
                        return False
        return True
    return False
