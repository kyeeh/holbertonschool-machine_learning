#!/usr/bin/env python3
"""
Convolutions and Pooling
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images

    images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images

        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images

    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for
    the convolution

        kh is the height of the kernel
        kw is the width of the kernel

    Returns: a numpy.ndarray containing the convolved images
    """
    kh, kw = kernel.shape
    m, h, w = images.shape
    pixy = h - kh + 1
    pixx = w - kw + 1
    cvv_img = np.zeros((m, pixy, pixx))
    for i in range(pixy):
        for j in range(pixx):
            cvv_img[:, i, j] = (images[:, i: i + kh, j: j + kw] *
                                kernel).sum(axis=(1, 2))
    return cvv_img
