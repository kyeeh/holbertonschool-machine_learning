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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression (lrc)

        Y: It's a numpy.ndarray with shape (1, m) that contains the correct
           labels for the input data
        A: It's a numpy.ndarray with shape (1, m) containing the activated
           output of the neuron for each example
        To avoid division by zero errors used is 1.0000001 - A instead of 1-A

        Returns the cost
        """
        lrc = np.sum(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))
        return -lrc / Y.shape[1]

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions (prd)

        X: It's a numpy.ndarray with shape (nx, m) that contains the input data
           nx: It's the number of input features to the neuron
           m: It's the number of examples
        Y: It's a numpy.ndarray with shape (1, m) that contains the correct
           labels for the input data
        Returns the neuron’s prediction (prd) and the cost of the network,
        respectively

        - The prediction should be a numpy.ndarray with shape (1, m) containing
          the predicted labels for each example
        - The label values should be 1 if the output of the network is >= 0.5
          and 0 otherwise
        """
        self.forward_prop(X)
        lrc = self.cost(Y, self.__A2)
        prd = np.where(self.__A2 >= 0.5, 1, 0)
        return (prd, lrc)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron

        X: It's a numpy.ndarray with shape (nx, m) that contains the input data
           nx: It's the number of input features to the neuron
           m: It's the number of examples
        Y: It's a numpy.ndarray with shape (1, m) that contains the correct
           labels for the input data
        A1: it's the output of the hidden layer
        A2: it's the predicted output
        alpha: It's the learning rate

        Updates the private attributes __W1, __b1, __W2, and __b2
        """
        m = A1.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        dz1a = np.matmul(self.__W2.T, dz2)
        dz1b = A1 * (1 - A1)
        dz1 = dz1a * dz1b
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron

        X: It's a numpy.ndarray with shape (nx, m) that contains the input data
           nx: It's the number of input features to the neuron
           m: It's the number of examples
        Y: It's a numpy.ndarray with shape (1, m) that contains the correct
           labels for the input data
        iterations: it's the number of iterations to train over
        alpha: It's the learning rate
        Updates the private attributes __W1, __b1, __A1, __W2, __b2, and __A2

        Returns the evaluation of the training data after iterations of
        training have occurred
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        return self.evaluate(X, Y)
