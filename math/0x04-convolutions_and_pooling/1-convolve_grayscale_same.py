#!/usr/bin/env python3
"""
Convolutions and Pooling
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

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
    padw = int(kw / 2) if kw % 2 == 0 else int((kw - 1) / 2)
    padh = int(kh / 2) if kh % 2 == 0 else int((kh - 1) / 2)
    padded = np.pad(images, ((0,), (padh,), (padw,)))
    cvv_img = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            cvv_img[:, i, j] = (padded[:, i: i + kh, j: j + kw] *
                                kernel).sum(axis=(1, 2))
    return cvv_img
