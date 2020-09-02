#!/usr/bin/env python3
"""
Keras Module
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network

    network is the network model to test
    data is the input data to test the model with
    labels are the correct one-hot labels of data
    verbose is a boolean that determines if output should be printed during
    the testing process

    Returns: the prediction for the data
    """
    return network.predict(x=data, verbose=verbose)
