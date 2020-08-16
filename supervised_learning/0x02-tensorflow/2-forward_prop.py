#!/usr/bin/env python3
"""
Tensorflow Module
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network

    x is the placeholder for the input data
    layer_sizes is a list with the number of nodes in each layer of the network
    activations is a list with the act. functions for each layer of the network

    Returns: the prediction of the network in tensor form
    """
    prd = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        prd = create_layer(prd, layer_sizes[i], activations[i])
    return prd
