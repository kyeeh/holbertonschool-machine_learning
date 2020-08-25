#!/usr/bin/env python3
"""
Error Analysis Module
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix

    confusion is a confusion numpy.ndarray of shape (classes, classes) where
    row indices represent the correct labels and column indices represent the
    predicted label

    classes is the number of classes

    Returns: a numpy.ndarray of shape (classes,) containing the sensitivity of
    each class
    """
    tp = np.diagonal(confusion)
    tp_fn = np.sum(confusion, axis=1)
    return tp / tp_fn
