#!/usr/bin/env python3
"""
Module with function to adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    Function to adds two arrays element-wise
    Returns the a new array with the result
    """
    result = []
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            result.append(arr1[i] + arr2[i])
        return result
    return None
