#!/usr/bin/env python3
"""
Convolutions and Pooling
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
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

    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’

        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
        the image should be padded with 0’s

    stride is a tuple of (sh, sw)

        sh is the stride for the height of the image
        sw is the stride for the width of the image

    Returns: a numpy.ndarray containing the convolved images
    """
    sh, sw = stride
    kh, kw = kernel.shape
    m, h, w = images.shape
    if padding == 'same':
        padw = int((((w - 1) * sw + kw - w) / 2) + 1)
        padh = int((((h - 1) * sh + kh - h) / 2) + 1)
    elif type(padding) is tuple:
        padh, padw = padding
    else:
        padh, padw = (0, 0)
    pixy = (((h + (padh * 2) - kh) // sh) + 1)
    pixx = (((w + (padw * 2) - kw) // sw) + 1)
    cvv_img = np.zeros((m, pixy, pixx))
    padded = np.pad(images, ((0,), (padh,), (padw,)))
    for i in range(pixy):
        for j in range(pixx):
            cvv_img[:, i, j] = (padded[:, (i * sh): (i * sh) + kh,
                                          (j * sw): (j * sw) + kw] *
                                kernel).sum(axis=(1, 2))
    return cvv_img
