#!/usr/bin/env python3
"""
Error Analysis Module
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix

    confusion is a confusion numpy.ndarray of shape (classes, classes) where
    row indices represent the correct labels and column indices represent the
    predicted label

    classes is the number of classes

    Returns: a numpy.ndarray of shape (classes,) containing the precision of
    each class
    """
    tp = np.diagonal(confusion)
    fp_tn = np.sum(confusion, axis=0)
    return tp / fp_tn
