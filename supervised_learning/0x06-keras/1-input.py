#!/usr/bin/env python3
"""
Keras Module
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library

    nx is the number of input features to the network
    layers is a list containing the number of nodes in each layer
    activations is a list containing the functions used for each layer
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout

    Returns: the keras model
    """
    inft = K.Input(shape=(nx,))
    x = K.layers.Dense(units=layers[0], activation=activations[0],
                       kernel_regularizer=K.regularizers.l2(lambtha))(inft)
    for i in range(1, len(layers)):
        x = K.layers.Dense(units=layers[i], activation=activations[i],
                           kernel_regularizer=K.regularizers.l2(lambtha),
                           input_shape=(nx,))(x)
        x = K.layers.Dropout(1 - keep_prob)(x)
    km = K.models.Model(inputs=inft, outputs=x)
    return km
