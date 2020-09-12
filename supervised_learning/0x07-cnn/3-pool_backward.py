#!/usr/bin/env python3
"""
CNN Module
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network

    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the output of the pooling layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c is the number of channels in the output

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer

    kernel_shape is a tuple of (kh, kw) containing the size of the kernel for
    the pooling
        kh is the kernel height
        kw is the kernel width

    mode is a string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively

    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width

    Returns: the partial derivatives with respect to the previous layer
    (dA_prev)
    """
    sh, sw = stride
    kh, kw = kernel_shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    dm, h_new, w_new, c_new = dA.shape
    dA_prev = np.zeros(A_prev.shape)
    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                jsh = j * sh
                ksw = k * sw
                for ll in range(c_new):
                    pool = A_prev[i, jsh: jsh + kh, ksw: ksw + kw, ll]
                    if mode == 'max':
                        maxp = np.amax(pool)
                        mask = np.zeros(kernel_shape)
                        np.place(mask, pool == maxp, 1)
                        dA_prev[i, jsh: jsh + kh, ksw: ksw + kw, ll] += \
                            mask * dA[i, j, k, ll]
                    else:
                        mask = np.ones(kernel_shape)
                        dA_prev[i, jsh: jsh + kh, ksw: ksw + kw, ll] += \
                            mask * dA[i, j, k, ll] / kh / kw
    return dA_prev
