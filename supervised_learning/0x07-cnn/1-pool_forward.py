#!/usr/bin/env python3
"""
CNN Module
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel for
    the pooling
        kh is the kernel height
        kw is the kernel width
    mode is a string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively
    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width

    Returns: the output of the pooling layer
    """
    sh, sw = stride
    kh, kw = kernel_shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    if mode == 'max':
        pool = np.max
    else:
        pool = np.average
    pixy = (((h_prev - kh) // sh) + 1)
    pixx = (((w_prev - kw) // sw) + 1)
    cvv_img = np.zeros((m, pixy, pixx, c_prev))
    for i in range(pixy):
        for j in range(pixx):
            cvv_img[:, i, j, :] = pool(A_prev[:, (i * sh): (i * sh) + kh,
                                              (j * sw): (j * sw) + kw],
                                       axis=(1, 2))
    return cvv_img
