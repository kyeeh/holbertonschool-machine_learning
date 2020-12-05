#!/usr/bin/env python3
"""
RNN module
"""
import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """
        class constructor
        - i: dimensionality of the data
        - h: dimensionality of the hidden state
        - o: dimensionality of the outputs
        Creates the public instance attributes Whf, Whb, Wy,
        bhf, bhb, by that represent the weights and biases of the cell
            - Whf and bhf are for the hidden states in the
            forward direction
            - Whb and bhb are for the hidden states in the
            backward direction
            - Wy and by are for the outputs
        """
        self.Whf = np.random.randn(h + i, h)
        self.Whb = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h * 2, o)

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ Method that calculates the hidden state in the forward
            direction for one time step
        Args:
            x_t - is a numpy.ndarray of shape (m, i) that contains the data
                input for the cell
            m - is the batch size for the data
            h_prev - is a numpy.ndarray of shape (m, h) containing the previous
                hidden state
        Returns:
            h_next - the next hidden state
        """
        h_next = np.tanh(np.matmul(np.hstack((h_prev, x_t)), self.Whf)
                         + self.bhf)
        return h_next
