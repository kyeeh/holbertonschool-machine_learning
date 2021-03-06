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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        lys = layers
        randn = np.random.randn
        for i in range(len(lys)):
            if (type(lys[i]) is not int) or (lys[i] <= 0):
                raise TypeError('layers must be a list of positive integers')
            if i == 0:
                self.weights['W1'] = randn(lys[i], nx) * np.sqrt(2 / nx)
            else:
                k = "W{}".format(i + 1)
                h = i - 1
                self.weights[k] = randn(lys[i], lys[h]) * np.sqrt(2 / lys[h])
            self.weights["b{}".format(i + 1)] = np.zeros((lys[i], 1))

    @property
    def L(self):
        """
        Getter method for number of layers
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter method for all intermediary values of the network
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter method for all weights and biased of the network
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron using a sigmoid
        activation function

        X: It's a numpy.ndarray with shape (nx,m) that contains the input data
          - nx: is the number of input features to the neuron
          - m: It's the number of examples
        Updates the private attribute __cache
          - The activated outputs of each layer should be saved in the __cache
            dictionary using the key A{l} where {l} is the hidden layer the
            activated output belongs to X should be saved to the cache
            dictionary using the key A0
        All neurons should use a sigmoid activation function

        Returns the output of the neural network and the cache, respectively
        """
        self.__cache['A0'] = X
        for i in range(self.L):
            a = self.__cache['A{}'.format(i)]
            b = self.weights['b{}'.format(i + 1)]
            w = self.weights['W{}'.format(i + 1)]
            fn = np.matmul(w, a) + b
            self.__cache['A{}'.format(i + 1)] = 1 / (1 + np.exp(-fn))
        return (self.__cache['A{}'.format(self.__L)], self.__cache)

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
        A = self.__cache['A{}'.format(self.__L)]
        lrc = self.cost(Y, A)
        prd = np.where(A >= 0.5, 1, 0)
        return (prd, lrc)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the DNN

        Y: It's a numpy.ndarray with shape (1, m) that contains the correct
           labels for the input data
        cache: it's a dictionary containing all the intermediary values of the
           network
        alpha: It's the learning rate

        Updates the private attribute __weights
        """
        m = Y.shape[1]
        last = self.__weights.copy()
        for i in range(self.__L, 0, -1):
            A = self.__cache['A{}'.format(i)]
            if i == self.__L:
                dz = A - Y
            else:
                W = 'W{}'.format(i + 1)
                dz = np.matmul(last[W].T, dz) * A * (1 - A)
            dw = np.matmul(dz, self.__cache['A{}'.format(i - 1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            b = self.__weights['b{}'.format(i)]
            w = self.__weights['W{}'.format(i)]
            self.__weights['b{}'.format(i)] = b - alpha * db
            self.__weights['W{}'.format(i)] = w - alpha * dw

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the DNN

        X: It's a numpy.ndarray with shape (nx, m) that contains the input data
           nx: It's the number of input features to the neuron
           m: It's the number of examples
        Y: It's a numpy.ndarray with shape (1, m) that contains the correct
           labels for the input data
        iterations: it's the number of iterations to train over
        alpha: It's the learning rate
        Updates the private attributes __weights and __cache

        Returns the evaluation of the training data after n iterations
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
            self.gradient_descent(Y, self.__cache, alpha)
        return self.evaluate(X, Y)
