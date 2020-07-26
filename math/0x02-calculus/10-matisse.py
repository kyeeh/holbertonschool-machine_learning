#!/usr/bin/env python3


def poly_derivative(poly):
    drv = [0]
    if type(poly) == list:
        if len(poly) > 1:
            i = 2
            drv = [poly[1]]
            for coef in poly[2:]:
                drv.append(coef * i)
                i += 1
        return drv
    return None
