#!/usr/bin/env python3
"""
Error Analysis Module
"""
import numpy as np
precision = __import__('2-precision').precision
sensitivity = __import__('1-sensitivity').sensitivity


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix

    confusion is a confusion numpy.ndarray of shape (classes, classes) where
    row indices represent the correct labels and column indices represent the
    predicted label

    classes is the number of classes

    Returns: a numpy.ndarray of shape (classes,) containing the F1 score of
    each class
    """
    ppv = precision(confusion)
    tpr = sensitivity(confusion)
    return 2 * ppv * tpr / (ppv + tpr)
