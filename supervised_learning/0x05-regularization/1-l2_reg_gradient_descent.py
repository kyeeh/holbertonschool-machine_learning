#!/usr/bin/env python3
"""
Error Analysis Module
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization

    Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
        classes is the number of classes
        m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs of each layer of the neural network
    alpha is the learning rate
    lambtha is the L2 regularization parameter
    L is the number of layers of the network
    The neural network uses tanh activations on each layer except the last,
    which uses a softmax activation

    The weights and biases of the network should be updated in place
    """
    m = Y.shape[1]
    dz = [cache['A{}'.format(L)] - Y]
    for layer in range(L, 0, -1):
        A = cache['A{}'.format(layer - 1)]
        W = weights['W{}'.format(layer)]
        db = np.sum(dz[L - layer], axis=1, keepdims=True) / m
        dw = np.matmul(dz[L - layer], A.T) / m
        dzdx = dz.append(np.matmul(W.T, dz[L - layer]) * (1 - (A ** 2)))
        l2 = dw + (lambtha / m) * weights['W{}'.format(layer)]
        weights['W{}'.format(layer)] -= alpha * l2
        weights['b{}'.format(layer)] -= alpha * db
