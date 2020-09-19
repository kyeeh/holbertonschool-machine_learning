#!/usr/bin/env python3
"""
Deep CNN Module
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely Connected
    Convolutional Networks

    X is the output from the previous layer

    nb_filters is an integer representing the number of filters in X

    compression is the compression factor for the transition layer

    layers is the number of layers in the transition layer

    Your code should implement compression as used in DenseNet-C

    All weights should use he normal initialization

    All convolutions should be preceded by Batch Normalization and a rectified
    linear activation (ReLU), respectively

    Returns: The output of the transition layer and the number of filters
    within the output, respectively
    """
    init = K.initializers.he_normal(seed=None)

    trly = K.layers.BatchNormalization(axis=3)(X)
    trly = K.layers.Activation('relu')(trly)

    FC = int(nb_filters * compression)
    trly = K.layers.Conv2D(filters=FC,
                           kernel_size=1,
                           padding='same',
                           kernel_initializer=init)(trly)

    apool = K.layers.AveragePooling2D(padding='same', pool_size=2)(trly)

    return (apool, FC)
