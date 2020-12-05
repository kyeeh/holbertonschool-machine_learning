#!/usr/bin/env python3
"""
RNN module
"""
import numpy as np


class LSTMCell:
    """
    LSTM class
    """
    def __init__(self, i, h, o):
        """
        class constructor
        - i: dimensionality of the data
        - h: dimensionality of the hidden state
        - o: dimensionality of the outputs
        Creates the public instance attributes Wf, Wu, Wc, Wo,
        Wy, bf, bu, bc, bo, by that represent the weights and
        biases of the cell
            - Wf and bf are for the forget gate
            - Wu and bu are for the update gate
            - Wc and bc are for the intermediate cell state
            - Wo and bo are for the output gate
            - Wy and by are for the outputs
        """
        self.Wf = np.random.randn(h+i, h)
        self.Wu = np.random.randn(h+i, h)
        self.Wc = np.random.randn(h+i, h)
        self.Wo = np.random.randn(h+i, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, z):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        """Compute softmax values for each sets of scores in x"""
        e_z = np.exp(z)
        return e_z / e_z.sum(axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """ Method that performs forward propagation for one time step
        Args:
            x_t - is a numpy.ndarray of shape (m, i) that contains the data
                input for the cell
            m - is the batche size for the data
            h_prev - is a numpy.ndarray of shape (m, h) containing the previous
                hidden state
            c_prev - is a numpy.ndarray of shape (m, h) containing the previous
                cell state
        Returns: h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        fg = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)
        ug = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)
        cct = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        c_next = ug * cct + fg * c_prev
        ot = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)
        h_next = ot * np.tanh(c_next)

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y
