#!/usr/bin/env python3
"""
Derivative
"""


def poly_derivative(poly):
    """
    Derivative of polynomials
    poly is a list of coefficients representing a polynomial
        the index of the list represents the power of x that the coefficient belongs to
        Example: if [f(x) = x^3 + 3x +5] , poly is equal to [5, 3, 0, 1]
    C is an integer representing the integration constant
    If a coefficient is a whole number, it should be represented as an integer
    If poly or C are not valid, return None
    Return a new list of coefficients representing the integral of the polynomial
    The returned list should be as small as possible
    """
    if type(poly) == list and len(poly) > 0:
        if len(poly) > 1:
            drv = []
            for i in range(1, len(poly)):
                if isinstance(poly[i], (int, float)):
                    drv.append(poly[i] * i)
                else:
                    return None
            return drv
        return [0]
    return None
