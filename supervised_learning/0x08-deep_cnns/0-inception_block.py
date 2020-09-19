#!/usr/bin/env python3
"""
Deep CNN Module
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in Going Deeper with
    Convolutions (2014)

    A_prev is the output from the previous layer

    filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
    respectively:
        - F1 is the number of filters in the 1x1 convolution
        - F3R is the number of filters in the 1x1 convolution before
          the 3x3 convolution
        - F3 is the number of filters in the 3x3 convolution
        - F5R is the number of filters in the 1x1 convolution before
          the 5x5 convolution
        - F5 is the number of filters in the 5x5 convolution
        - FPP is the number of filters in the 1x1 convolution after
          the max pooling

    All convolutions inside the inception block should use a rectified linear
    activation (ReLU)

    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal(seed=None)

    cvv_F1 = K.layers.Conv2D(filters=F1,
                             kernel_size=1,
                             padding='same',
                             activation='relu',
                             kernel_initializer=init)(A_prev)

    cvv_F3R = K.layers.Conv2D(filters=F3R,
                              kernel_size=1,
                              padding='same',
                              activation='relu',
                              kernel_initializer=init)(A_prev)

    cvv_F3 = K.layers.Conv2D(filters=F3,
                             kernel_size=3,
                             padding='same',
                             activation='relu',
                             kernel_initializer=init)(cvv_F3R)

    cvv_F5R = K.layers.Conv2D(filters=F5R,
                              kernel_size=1,
                              padding='same',
                              activation='relu',
                              kernel_initializer=init)(A_prev)

    cvv_F5 = K.layers.Conv2D(filters=F5,
                             kernel_size=5,
                             padding='same',
                             activation='relu',
                             kernel_initializer=init)(cvv_F5R)

    pool = K.layers.MaxPool2D(padding='same',
                              pool_size=3,
                              strides=1)(A_prev)

    cvv_FPP = K.layers.Conv2D(filters=FPP,
                              kernel_size=1,
                              padding='same',
                              activation='relu',
                              kernel_initializer=init)(pool)

    output = K.layers.concatenate([cvv_F1, cvv_F3, cvv_F5, cvv_FPP])

    return output
