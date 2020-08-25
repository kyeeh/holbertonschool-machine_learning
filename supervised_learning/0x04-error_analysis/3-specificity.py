#!/usr/bin/env python3
"""
Error Analysis Module
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix

    confusion is a confusion numpy.ndarray of shape (classes, classes) where
    row indices represent the correct labels and column indices represent the
    predicted label

    classes is the number of classes

    Returns: a numpy.ndarray of shape (classes,) containing the specificity of
    each class
    """
    tp = np.diagonal(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    tn = np.sum(confusion) - (fp + fn + tp)
    return tn / (tn + fp)
