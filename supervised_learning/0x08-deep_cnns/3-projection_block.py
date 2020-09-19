#!/usr/bin/env python3
"""
Deep CNN Module
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds an projection block as described in Deep Residual Learning for Image
    Recognition (2015)

    A_prev is the output from the previous layer

    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution

    s is the stride of the first convolution in both the main path and the
    shortcut connection

    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear activation
    (ReLU), respectively.

    All weights should use he normal initialization

    Returns: the activated output of the projection block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=None)

    cvv_F11 = K.layers.Conv2D(filters=F11,
                              strides=s,
                              kernel_size=1,
                              padding='same',
                              kernel_initializer=init)(A_prev)

    cvv_F11 = K.layers.BatchNormalization(axis=3)(cvv_F11)
    cvv_F11 = K.layers.Activation('relu')(cvv_F11)

    cvv_F3 = K.layers.Conv2D(filters=F3,
                             kernel_size=3,
                             padding='same',
                             kernel_initializer=init)(cvv_F11)

    cvv_F3 = K.layers.BatchNormalization(axis=3)(cvv_F3)
    cvv_F3 = K.layers.Activation('relu')(cvv_F3)

    cvv_F12 = K.layers.Conv2D(filters=F12,
                              kernel_size=1,
                              padding='same',
                              kernel_initializer=init)(cvv_F3)

    cvv_F12 = K.layers.BatchNormalization(axis=3)(cvv_F12)

    cvp_F12 = K.layers.Conv2D(filters=F12,
                              strides=s,
                              kernel_size=1,
                              padding='same',
                              kernel_initializer=init)(A_prev)

    cvp_F12 = K.layers.BatchNormalization(axis=3)(cvp_F12)

    add = K.layers.Add()([cvv_F12, cvp_F12])
    output = K.layers.Activation('relu')(add)

    return output
