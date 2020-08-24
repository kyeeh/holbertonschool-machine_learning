#!/usr/bin/env python3
"""
Optimization Module
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum optimization

    alpha is the learning rate
    beta1 is the momentum weight
    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    v is the previous first moment of var

    Returns: the updated variable and the new moment, respectively
    """
    vdv = (beta1 * v) + ((1 - beta1) * grad)
    vup = var - (alpha * vdv)
    return vup, vdv
