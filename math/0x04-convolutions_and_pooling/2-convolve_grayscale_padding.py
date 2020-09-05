#!/usr/bin/env python3
"""
Convolutions and Pooling
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding

    images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images

        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images

    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for
    the convolution

        kh is the height of the kernel
        kw is the width of the kernel

    padding is a tuple of (ph, pw)

        ph is the padding for the height of the image
        pw is the padding for the width of the image
        the image should be padded with 0’s

    Returns: a numpy.ndarray containing the convolved images
    """
    padh, padw = padding
    kh, kw = kernel.shape
    m, h, w = images.shape
    pixy = h + 2 * padh - kh + 1
    pixx = w + 2 * padw - kw + 1
    cvv_img = np.zeros((m, pixy, pixx))
    padded = np.pad(images, ((0,), (padh,), (padw,)))
    for i in range(pixy):
        for j in range(pixx):
            cvv_img[:, i, j] = (padded[:, i: i + kh, j: j + kw] *
                                kernel).sum(axis=(1, 2))
    return cvv_img
