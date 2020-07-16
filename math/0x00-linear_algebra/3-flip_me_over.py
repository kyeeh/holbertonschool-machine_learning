#!/usr/bin/env python3
"""
Module with function to transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Function to transpose a Matrix
    Returns the transpose of a 2D matrix
    """
    return [[matrix[j][i] for j in range(len(matrix))]
            for i in range(len(matrix[0]))]
