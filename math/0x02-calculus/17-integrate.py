#!/usr/bin/env python3
"""
Integral of polynomials
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial:

    poly is a list of coefficients representing a polynomial
        the index of the list represents the power of x that the coefficient
            belongs to
        Example: if [f(x) = x^3 + 3x +5] , poly is equal to [5, 3, 0, 1]
    C is an integer representing the integration constant
    If a coefficient is a whole number, it should be represented as an integer
    If poly or C are not valid, return None
    Return a new list of coefficients representing the integral of the
        polynomial
    The returned list should be as small as possible

    """
    if type(poly) == list and len(poly) > 0 and type(C) == int:
        itg = [C]
        if len(poly) > 1:
            for i in range(1, len(poly)):
                if isinstance(poly[i], (int, float)):
                    coef = poly[i - 1] / i
                    if coef.is_integer():
                        itg.append(int(coef))
                    else:
                        itg.append(coef)
                else:
                    return None
        else:
            if poly[0] == 0:
                return itg
            else:
                return [C, poly[0]]
        coef = poly[len(poly) - 1] / len(poly)
        if coef.is_integer():
            itg.append(int(coef))
        else:
            itg.append(coef)
        for i in range(len(itg)):
            if (sum(itg[i:]) == 0):
                return itg[:i]
        return itg
    return None
