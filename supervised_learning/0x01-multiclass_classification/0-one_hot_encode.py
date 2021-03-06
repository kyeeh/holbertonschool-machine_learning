#!/usr/bin/env python3
"""
One-Hot Matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Y: it's a numpy.ndarray with shape (m,) containing numeric class labels
      - m: it's the number of examples
    classes: is the maximum number of classes found in Y

    Returns: a one-hot matrix (ohm) encoding of Y with shape (classes,m), or
    None on failure
    """
    if (len(Y) == 0) or (type(Y) != np.ndarray):
        return None
    if ((type(classes) is not int or classes < np.max(Y))):
        return None
    ax = np.arange(len(Y))
    ohm = np.zeros((classes, len(Y)))
    ohm[Y, ax] = 1
    return ohm
