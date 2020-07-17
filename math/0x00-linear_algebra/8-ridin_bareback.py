#!/usr/bin/env python3
"""
Module with function to performs matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
    Function to performs matrix multiplication
    Returns the a new matrix
    """
    result = []
    if len(mat1[0]) == len(mat2):
        for i in range(len(mat1)):
            aux_row = []
            for j in range(len(mat2[0])):
                aux = 0
                for k in range(len(mat2)):
                    aux += mat1[i][k] * mat2[k][j]
                aux_row.append(aux)
            result.append(aux_row)
        return result
    return None
