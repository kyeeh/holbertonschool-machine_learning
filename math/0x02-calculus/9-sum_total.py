#!/usr/bin/env python3
"""
Series
"""


def summation_i_squared(n):
    """
    Sum of i^2
    """
    if type(n) == int and n > 0:
        return int((n * (n + 1) * (2 * n + 1)) / 6)
    return None
