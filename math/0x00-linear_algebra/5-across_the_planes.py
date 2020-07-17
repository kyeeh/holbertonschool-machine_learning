#!/usr/bin/env python3
"""
Module with function to adds two Matrices
"""


def add_matrices2D(mat1, mat2):
    """
    Function to adds two matrices
    Returns the a new matrix with the result
    """
    if len(mat1) == len(mat2[0]):
        return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
                for i in range(len(mat2))]
    return None
