#!/usr/bin/env python3
"""
Module with function to adds two Matrices
"""


def add_matrices2D(mat1, mat2):
    """
    Function to adds two matrices
    Returns the a new matrix with the result
    """
    result = []
    if len(mat1) == len(mat2):
        for i in range(len(mat1)):
            if len(mat1[i]) == len(mat2[i]):
                aux = [mat1[i][j] + mat2[i][j] for j in range(len(mat2[i]))]
                result.append(aux)
            else:
                return None
        return result            
    return None
