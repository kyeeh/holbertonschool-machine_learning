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
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
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


def minor(matrix):
    """
    Calculates the minor matrix of a matrix

    matrix is a list of lists whose minor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
    matrix must be a non-empty square matrix

    Returns: the minor matrix of matrix
    """
    if (type(matrix) != list or len(matrix) == 0 or
       not all([type(m) == list for m in matrix])):
        raise TypeError('matrix must be a list of lists')
    col_size = [len(row) for row in matrix]
    if matrix == [[]]:
        raise ValueError('matrix must be a non-empty square matrix')
    if not all(len(matrix) == col for col in col_size):
        raise ValueError('matrix must be a non-empty square matrix')
    if len(matrix) == 1:
        return [[1]]

    minors_list = []
    for i in range(len(matrix)):
        minors = []
        for j in range(len(matrix)):
            minor = [row[:j] + row[j+1:]for row in (matrix[:i]+matrix[i+1:])]
            minors.append(determinant(minor))
        minors_list.append(minors)
    return minors_list


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix

    matrix is a list of lists whose cofactor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
    matrix must be a non-empty square matrix

    Returns: the cofactor matrix of matrix
    """
    minors = minor(matrix)
    cofact = minors.copy()
    for i in range(len(minors)):
        for j in range(len(minors)):
            cofact[i][j] = cofact[i][j] * (-1)**(i+j)
    return cofact


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix

    matrix is a list of lists whose cofactor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
    matrix must be a non-empty square matrix

    Returns: the adjugate matrix of matrix
    """
    cofact = cofactor(matrix)
    return [[row[i] for row in cofact] for i in range(len(cofact[0]))]


def inverse(matrix):
    """
    Calculates the adjugate matrix of a matrix

    matrix is a list of lists whose cofactor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
    matrix must be a non-empty square matrix

    Returns: the inverse of matrix, or None if matrix is singular.
    """
    dtrm = determinant(matrix)
    if dtrm == 0:
        return None

    adjt = adjugate(matrix)
    return [[n / dtrm for n in row] for row in adjt]
