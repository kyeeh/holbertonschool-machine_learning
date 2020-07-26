#!/usr/bin/env python3
"""
Series
"""


def summation_i_squared(n):
    """
    Write a function def summation_i_squared(n): that calculates sum_{i=1}^{n} i^2
    n is the stopping condition
    Return the integer value of the sum
    If n is not a valid number, return None
    You are not allowed to use any loops

    """
    if type(n) == int and n > 0:
        return int((n * (n + 1) * (2 * n + 1)) / 6)
    return None
