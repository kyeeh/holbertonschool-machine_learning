#!/usr/bin/env python3
"""
Tensorflow Module
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network

    loss is the loss of the network’s prediction
    alpha is the learning rate

    Returns: an operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train
