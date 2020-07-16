#!/usr/bin/env python3
"""
Module with functions to calculate the shape of a Matrix
"""


def matrix_shape(matrix):
    """
    Setup function to call the recursive version
    Returns the shape as a list of integers
    """
    shape = []
    return rec_matrix_shape(matrix, shape)


def rec_matrix_shape(matrix, shape):
    """
    Recursive function to calculate the shape of a Matrix
    Returns the shape as a list of integers
    """
    if type(matrix) == list:
        shape.append(len(matrix))
        rec_matrix_shape(matrix[0], shape)
    return shape
