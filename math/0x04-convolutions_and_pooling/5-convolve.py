#!/usr/bin/env python3
"""
Convolutions and Pooling
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels

    images is a numpy.ndarray with shape (m, h, w) containing multiple
    images

        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image

    kernels is a numpy.ndarray with shape (kh, kw, c, nc) containing the
    kernels for the convolution

        kh is the height of the kernel
        kw is the width of the kernel
        nc is the number of kernels

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
    kh, kw, c, nc = kernels.shape
    m, h, w, c = images.shape
    if padding == 'same':
        padw = int((((w - 1) * sw + kw - w) / 2) + 1)
        padh = int((((h - 1) * sh + kh - h) / 2) + 1)
    elif type(padding) is tuple:
        padh, padw = padding
    else:
        padh, padw = (0, 0)
    pixy = (((h + (padh * 2) - kh) // sh) + 1)
    pixx = (((w + (padw * 2) - kw) // sw) + 1)
    cvv_img = np.zeros((m, pixy, pixx, nc))
    padded = np.pad(images, ((0,), (padh,), (padw,), (0,)))
    for i in range(pixy):
        for j in range(pixx):
            for k in range(nc):
                cvv_img[:, i, j, k] = (padded[:, (i * sh): (i * sh) + kh,
                                              (j * sw): (j * sw) + kw] *
                                       kernels[:, :, :, k]).
                sum(axis=(1, 2, 3))
    return cvv_img
