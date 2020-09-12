#!/usr/bin/env python3
"""
CNN Module
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network

    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the unactivated output of the
    convolutional layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer

    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
    kernels for the convolution
        kh is the filter height
        kw is the filter width
        c_prev is the number of channels in the previous layer
        c_new is the number of channels in the output

    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution

    padding is a string that is either same or valid, indicating the type of
    padding used

    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width

    Returns: the partial derivatives with respect to the previous layer
    (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    sh, sw = stride
    kh, kw, c, c_new = W.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    d, h_new, w_new, _ = dZ.shape
    if padding == 'same':
        padw = int((((w_prev - 1) * sw + kw - w_prev) / 2) + 1)
        padh = int((((h_prev - 1) * sh + kh - h_prev) / 2) + 1)
    else:
        padh, padw = (0, 0)
    A_prev = np.pad(A_prev, ((0,), (padh,), (padw,), (0,)), constant_values=0,
                    mode='constant')
    dW = np.zeros(W.shape)
    dA = np.zeros(A_prev.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                jsh = j * sh
                ksw = k * sw
                for ll in range(c_new):
                    dW[:, :, :, ll] += A_prev[i, jsh: jsh + kh,
                                              ksw: ksw + kw, :] * \
                        dZ[i, j, k, ll]
                    dA[i, jsh: jsh + kh, ksw: ksw + kw, :] += \
                        dZ[i, j, k, ll] * W[:, :, :, ll]
    if padding == 'same':
        dA = dA[:, padh: -padh, padw: -padw, :]
    return dA, dW, db
