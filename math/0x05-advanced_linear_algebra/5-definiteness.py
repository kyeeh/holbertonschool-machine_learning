#!/usr/bin/env python3
"""
Advanced Linear Algebra Module
"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.
    matrix is a numpy.ndarray of shape (n, n) whose definiteness should be
    calculated
    If matrix is not a numpy.ndarray, raise a TypeError with the message
    matrix must be a numpy.ndarray
    If matrix is not a valid matrix, return None

    Return: the string Positive definite, Positive semi-definite, Negative
    semi-definite, Negative definite, or Indefinite if the matrix is positive
    definite, positive semi-definite, negative semi-definite, negative
    definite of indefinite, respectively

    If matrix does not fit any of the above categories, return None
    """
    if type(matrix) != np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')
    if len(matrix.shape) == 1 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.linalg.eig(matrix):
        return None
    if not (matrix.transpose() == matrix).all():
        return None

    try:
        eig = np.linalg.eigvals(matrix)
        if all(eig > 0):
            return 'Positive definite'
        if all(eig >= 0):
            return 'Positive semi-definite'
        if all(eig < 0):
            return 'Negative definite'
        if all(eig <= 0):
            return 'Negative semi-definite'
        return 'Indefinite'
    except Exception:
        return None
