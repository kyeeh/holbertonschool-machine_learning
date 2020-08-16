#!/usr/bin/env python3
"""
Tensorflow Module
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction

    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions

    Returns: a tensor containing the decimal accuracy of the prediction
    """
    val = tf.argmax(y, axis=1)
    prd = tf.argmax(y_pred, axis=1)
    compare = tf.equal(val, prd)
    accuracy = tf.reduce_mean(tf.cast(compare, tf.float32))
    return accuracy
