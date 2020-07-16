#!/usr/bin/env python3
"""
Module with function to adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    Function to adds two arrays element-wise
    Returns the a new array with the result
    """
    if len(arr1) == len(arr2):
        return [arr1[i] + arr2[i] for i in range(len(arr1))]
    return None
