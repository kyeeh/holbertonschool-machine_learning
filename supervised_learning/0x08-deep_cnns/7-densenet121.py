#!/usr/bin/env python3
"""
Deep CNN Module
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in Densely Connected
    Convolutional Networks

    growth_rate is the growth rate
    compression is the compression factor
    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization and a rectified
    linear activation (ReLU), respectively
    All weights should use he normal initialization

    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)

    cvv1 = K.layers.BatchNormalization(axis=3)(X)
    cvv1 = K.layers.Activation('relu')(cvv1)
    cvv1 = K.layers.Conv2D(filters=2*growth_rate,
                           kernel_size=7,
                           padding='same',
                           strides=2,
                           kernel_initializer=init)(cvv1)

    mpool1 = K.layers.MaxPool2D(padding='same',
                                pool_size=3,
                                strides=2)(cvv1)

    layers = [12, 24, 16]
    dnsb, nb_filters = dense_block(mpool1, 2*growth_rate, growth_rate, 6)
    for lyl in layers:
        trly, nb_filters = transition_layer(dnsb, nb_filters, compression)
        dnsb, nb_filters = dense_block(trly, nb_filters, growth_rate, lyl)

    apool = K.layers.AveragePooling2D(padding='same', pool_size=7)(dnsb)

    output = K.layers.Dense(1000, activation='softmax')(apool)
    model = K.models.Model(inputs=X, outputs=output)
    return model
