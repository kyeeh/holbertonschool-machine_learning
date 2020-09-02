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
    km = K.Sequential()
    for i in range(len(layers)):
        km.add(K.layers.Dense(units=layers[i], activation=activations[i],
                              kernel_regularizer=K.regularizers.l2(lambtha),
                              input_shape=(nx,)))
    km.add(K.layers.Dropout(1 - keep_prob))
    return km
