#!/usr/bin/env python3
"""
Module with functions to performs element-wise operations
"""


def np_elementwise(mat1, mat2):
    """
    addition, subtraction, multiplication, and division
    Returns the new matrix
    """
    return(mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
