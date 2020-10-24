#!/usr/bin/env python3
"""
Advanced Linear Algebra Module
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix.

    matrix is a list of lists whose determinant should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists
    If matrix is not square, raise a ValueError with the message matrix must
    be a square matrix
    The list [[]] represents a 0x0 matrix

    Returns: the determinant of matrix
    """
    if (type(matrix) is not list or len(matrix) == 0 or
       not all([type(m) == list for m in matrix])):
        raise TypeError('matrix must be a list of lists')
    if len(matrix[0]) != len(matrix):
        raise ValueError("matrix must be a square matrix")
    if matrix == [[]]:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return ((matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]))

    dtrm = 0
    for i, n in enumerate(matrix[0]):
        mtrx = [[m[n] for n in range(len(m)) if n != i] for m in matrix[1:]]
        dtrm += (n * (-1)**i * determinant(mtrx))
    return dtrm
