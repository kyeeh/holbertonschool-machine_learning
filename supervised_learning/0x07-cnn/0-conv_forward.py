#!/usr/bin/env python3
"""
CNN Module
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network

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
    activation is an activation function applied to the convolution
    padding is a string that is either same or valid, indicating the type of
    padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width

    Returns: the output of the convolutional layer
    """
    sh, sw = stride
    kh, kw, c, c_new = W.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    if padding == 'same':
        padw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
        padh = int(((h_prev - 1) * sh + kh - h_prev) / 2)
    else:
        padh, padw = (0, 0)
    pixy = (((h_prev + (padh * 2) - kh) // sh) + 1)
    pixx = (((w_prev + (padw * 2) - kw) // sw) + 1)
    cvv_img = np.zeros((m, pixy, pixx, c_new))
    A_prev = np.pad(A_prev, ((0,), (padh,), (padw,), (0,)), constant_values=0,
                    mode='constant')
    for i in range(pixy):
        for j in range(pixx):
            for k in range(c_new):
                cvv_img[:, i, j, k] = (A_prev[:, (i * sh): (i * sh) + kh,
                                              (j * sw): (j * sw) + kw] *
                                       W[:, :, :, k]).sum(axis=(1, 2, 3))
                cvv_img[:, i, j, k] = activation(cvv_img[:, i, j, k] +
                                                 b[0, 0, 0, k])
    return cvv_img
