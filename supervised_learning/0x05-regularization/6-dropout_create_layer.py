#!/usr/bin/env python3
"""
Error Analysis Module
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization using
    gradient descent

    Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
        classes is the number of classes
        m is the number of data points

    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs and dropout masks of each layer of
    the neural network
    alpha is the learning rate
    keep_prob is the probability that a node will be kept
    L is the number of layers of the network
    All layers use thetanh activation function except the last, which uses
    the softmax activation function

    The weights of the network should be updated in place
    """
    m = Y.shape[1]
    dz = [cache['A{}'.format(L)] - Y]
    for layer in range(L, 0, -1):
        A = cache['A{}'.format(layer - 1)]
        W = weights['W{}'.format(layer)]
        dw = np.matmul(dz[L - layer], A.T) / m
        db = np.sum(dz[L - layer], axis=1, keepdims=True) / m
        if layer > 1:
            rglz = (1 - (A ** 2)) * (cache['D' + str(layer - 1)] / keep_prob)
            dzdx = dz.append(np.matmul(W.T, dz[L - layer]) * rglz)
        weights['W{}'.format(layer)] -= alpha * dw
        weights['b{}'.format(layer)] -= alpha * db
