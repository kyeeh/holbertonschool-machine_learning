"""
RNN module
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    performs forward propagation for a deep RNN:
    - rnn_cells: list of RNNCell instances of length l that will
    be used for the forward propagation
        - l: number of layers
    - X: data to be used, given as a numpy.ndarray of shape (t, m, i)
        - t: maximum number of time steps
        - m: batch size
        - i: dimensionality of the data
    - h_0: initial hidden state, given as a numpy.ndarray
    of shape (l, m, h)
        - h: dimensionality of the hidden state
    Returns: H, Y
        - H: numpy.ndarray containing all of the hidden states
        - Y: numpy.ndarray containing all of the outputs
    """
    T, m, i = X.shape
    L, _, h = h_0.shape

    H = np.zeros((T + 1, L, m, h))
    H[0] = h_0
    Y = []
    for t in range(T):
        aux = X[t]
        for lay in range(L):
            h_n, y = rnn_cells[lay].forward(H[t, lay], aux)
            H[t + 1, lay] = h_n
            aux = h_n
        Y.append(y)
    return H, np.array(Y)
