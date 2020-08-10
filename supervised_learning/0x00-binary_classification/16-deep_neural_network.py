#!/usr/bin/env python3
"""
Deep Neural Network Class
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Constructor method
        ------------------

        nx: it's the number of input features to the neuron
        layers: it's a list representing the number of nodes in each layer of
            the network
        L: The number of layers in the neural network.
        cache: A dictionary to hold all intermediary values of the network.
            Upon instantiation, it should be set to an empty dictionary
        weights: A dictionary to hold all weights and biased of the network.
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        lys = layers
        randn = np.random.randn
        for i in range(len(lys)):
            if (type(lys[i]) is not int) and (lys[i] <= 0):
                raise TypeError('layers must be a list of positive integers')
            if i == 0:
                self.weights['W1'] = randn(lys[i], nx) * np.sqrt(2 / nx)
            else:
                k = "W{}".format(i + 1)
                h = i - 1
                self.weights[k] = randn(lys[i], lys[h]) * np.sqrt(2 / lys[h])
            self.weights["b{}".format(i + 1)] = np.zeros((lys[i], 1))
