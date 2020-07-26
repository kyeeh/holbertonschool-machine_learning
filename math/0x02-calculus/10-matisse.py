#!/usr/bin/env python3
"""
Derivative
"""


def poly_derivative(poly):
    """
    Derivative of polynomials
    """
    if type(poly) == list and len(poly) > 0:
        if len(poly) > 1:
            drv = []
            for i in range(len(poly)):
                if isinstance(poly[i], (int, float)):
                    drv.append(poly[i] * i)
                else:
                    return None
            return drv[1:]
        return [0]
    return None
