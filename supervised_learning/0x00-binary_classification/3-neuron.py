#!/usr/bin/env python3
"""
Neuron Class
"""
import numpy as np


class Neuron:
    """
    Methods and attributes of Neuron class
    """

    def __init__(self, nx):
        """
        Constructor method
        ------------------

        nx: is the number of input features to the neuron
        W: The weights vector for the neuron. Upon instantiation, it should be
           initialized using a random normal distribution.
        b: The bias for the neuron. Upon instantiation, it should be
           initialized to 0.
        A: The activated output of the neuron (prediction). Upon
           instantiation, it should be initialized to 0.
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter method for weights
        """
        return self.__W

    @property
    def b(self):
        """
        Getter method for bias
        """
        return self.__b

    @property
    def A(self):
        """
        Getter method for activated output
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron using a sigmoid
        activation function

        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        fn is the neuron function applying weigths to input data + bias, it
        calculates the input for sigmoid activation function
        """
        fn = np.matmul(self.__W, X) + self.__b
        self.__A = 1. / (1 + np.exp(-fn))
        return self.__A
