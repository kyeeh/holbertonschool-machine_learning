#!/usr/bin/env python3
"""
Optimization Module
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm

    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero
    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    s is the previous second moment of var

    Returns: the updated variable and the new moment, respectively
    """
    vdv = (beta1 * v) + ((1 - beta1) * grad)
    sdv = (beta2 * s) + ((1 - beta2) * (grad ** 2))

    vdvc = vdv / (1 - (beta1 ** t))
    sdvc = sdv / (1 - (beta2 ** t))

    vup = var - ((alpha * vdvc) / ((sdvc ** (1/2)) + epsilon))
    return vup, vdv, sdv
