#!/usr/bin/env python3
"""
Neural Network Class
"""
import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer to do binary classification
    """

    def __init__(self, nx, nodes):
        """
        Constructor method
        ------------------

        nx: it's the number of input features to the neuron
        nodes: it's the number of nodes found in the hidden layer
        W1: The weights vector for the hidden layer. Upon instantiation, it
            should be initialized using a random normal distribution.
        b1: The bias for the hidden layer. Upon instantiation, it should be
            initialized with 0’s.
        A1: The activated output for the hidden layer. Upon instantiation, it
            should be initialized to 0.
        W2: The weights vector for the output neuron. Upon instantiation, it
            should be initialized using a random normal distribution.
        b2: The bias for the output neuron. Upon instantiation, it should be
            initialized to 0.
        A2: The activated output for the output neuron (prediction). Upon
            instantiation, it should be initialized to 0.
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter method for weights
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter method for bias
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter method for activated output
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter method for weights
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter method for bias
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter method for activated output
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron using a sigmoid
        activation function

        X: It's a numpy.ndarray with shape (nx,m) that contains the input data
           nx is the number of input features to the neuron
        m: It's the number of examples
        fn: It's the neuron function applying weigths to input data + bias, it
            calculates the input for sigmoid activation function

        Returns the private attributes __A1 and __A2, respectively
        """
        fn1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-fn1))
        fn2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-fn2))
        return self.__A1, self.__A2
