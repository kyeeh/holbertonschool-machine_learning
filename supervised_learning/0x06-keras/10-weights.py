#!/usr/bin/env python3
"""
Keras Module
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Saves a model’s weights

    network is the model whose weights should be saved
    filename is the path of the file that the weights should be saved to
    save_format is the format in which the weights should be saved

    Returns: None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    Loads a model’s weights

    filename is the path of the file that the model should be loaded from

    Returns: the loaded model
    """
    network.load_weights(filename)
    return None
