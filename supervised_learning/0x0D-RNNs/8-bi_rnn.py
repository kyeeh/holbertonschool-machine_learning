#!/usr/bin/env python3
"""
RNN module
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    performs forward propagation for a bidirectional RNN:
    - bi_cell: instance of BidirectinalCell that will be used
    for the forward propagation
    - X: data to be used, given as a numpy.ndarray of shape (t, m, i)
        - t: maximum number of time steps
        - m: batch size
        - i: dimensionality of the data
    - h_0: initial hidden state in the forward direction,
    given as a numpy.ndarray of shape (m, h)
        - h: dimensionality of the hidden state
    - h_t: initial hidden state in the backward direction,
    given as a numpy.ndarray of shape (m, h)
    Returns: H, Y
        - H: numpy.ndarray containing all of the concatenated hidden states
        - Y: numpy.ndarray containing all of the outputs
    """
    T, _, _ = X.shape
    m, h = h_0.shape

    H_n = np.zeros((T, m, h))
    H_p = np.zeros((T, m, h))

    H_n[0] = h_0
    H_p[T - 1] = h_t

    h_next = h_0
    h_prev = h_t

    for t in range(T):
        h_next = bi_cell.forward(h_next, X[t])
        h_prev = bi_cell.backward(h_prev, X[-t - 1])
        H_n[t] = h_next
        H_p[-t - 1] = h_prev

    H = np.concatenate((H_n, H_p), axis=-1)
    Y = bi_cell.output(H)
    return H, Y
