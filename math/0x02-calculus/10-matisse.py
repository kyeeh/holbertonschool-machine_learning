#!/usr/bin/env python3
"""
Derivative
"""


def poly_derivative(poly):
    """
    Derivative of polynomials
    """
    drv = [0]
    if type(poly) == list or len(poly) == 0:
        if len(poly) > 1:
            i = 2
            drv = [poly[1]]
            for coef in poly[2:]:
                drv.append(coef * i)
                i += 1
        return drv
    return None