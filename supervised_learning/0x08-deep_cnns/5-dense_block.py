#!/usr/bin/env python3
"""
Deep CNN Module
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected
    Convolutional Networks

    X is the output from the previous layer

    nb_filters is an integer representing the number of filters in X

    growth_rate is the growth rate for the dense block

    layers is the number of layers in the dense block

    You should use the bottleneck layers used for DenseNet-B

    All weights should use he normal initialization

    All convolutions should be preceded by Batch Normalization and a rectified
    linear activation (ReLU), respectively

    Returns: The concatenated output of each layer within the Dense Block and
    the number of filters within the concatenated outputs, respectively
    """
    init = K.initializers.he_normal(seed=None)

    for i in range(layers):
        cvv1 = K.layers.BatchNormalization(axis=3)(X)
        cvv1 = K.layers.Activation('relu')(cvv1)

        cvv1 = K.layers.Conv2D(filters=4*growth_rate,
                               kernel_size=1,
                               padding='same',
                               kernel_initializer=init)(cvv1)

        cvv2 = K.layers.BatchNormalization(axis=3)(cvv1)
        cvv2 = K.layers.Activation('relu')(cvv2)

        cvv2 = K.layers.Conv2D(filters=growth_rate,
                               kernel_size=3,
                               padding='same',
                               kernel_initializer=init)(cvv2)

        X = K.layers.concatenate([X, cvv2])
        nb_filters += growth_rate

    return (X, nb_filters)
