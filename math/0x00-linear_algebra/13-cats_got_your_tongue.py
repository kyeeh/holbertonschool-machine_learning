#!/usr/bin/env python3
"""
Module with function to Concatenate two matrices
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    Returns the new matrix
    """
    return np.concatenate((mat1, mat2), axis)
