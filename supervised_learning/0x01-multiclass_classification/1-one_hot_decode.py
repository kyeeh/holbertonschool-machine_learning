#!/usr/bin/env python3
"""
One-Hot Matrix
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    one_hot: it's a one-hot encoded numpy.ndarray with shape (classes, m)
      - classes: is the maximum number of classes
      - m: is the number of examples

    Returns: a numpy.ndarray with shape (m, ) containing the numeric labels
    for each example, or None on failure
    """
    if isinstance(one_hot, np.ndarray) and len(one_hot.shape) == 2:
        return np.argmax(one_hot, axis=0)
    return None
