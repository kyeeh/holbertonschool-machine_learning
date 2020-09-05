#!/usr/bin/env python3
"""
Convolutions and Pooling
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images

    images is a numpy.ndarray with shape (m, h, w) containing multiple
    images

        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image

    kernel_shape is a tuple of (kh, kw) containing the kernel shape for the
    pooling

        kh is the height of the kernel
        kw is the width of the kernel

    stride is a tuple of (sh, sw)

        sh is the stride for the height of the image
        sw is the stride for the width of the image

    mode indicates the type of pooling

        max indicates max pooling
        avg indicates average pooling

    Returns: a numpy.ndarray containing the convolved images
    """
    sh, sw = stride
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    if mode == 'max':
        pool = np.max
    else:
        pool = np.average
    pixy = (((h - kh) // sh) + 1)
    pixx = (((w - kw) // sw) + 1)
    cvv_img = np.zeros((m, pixy, pixx, c))
    for i in range(pixy):
        for j in range(pixx):
            cvv_img[:, i, j, :] = pool(images[:, (i * sh): (i * sh) + kh,
                                              (j * sw): (j * sw) + kw],
                                       axis=(1, 2))
    return cvv_img
