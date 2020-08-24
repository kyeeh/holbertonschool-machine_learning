#!/usr/bin/env python3
"""
Optimization Module
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow

    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created

    activation is the activation function that should be used on the output of
    the layer

    you should use the tf.layers.Dense layer as the base layer with kernal
    initializer tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    your layer should incorporate two trainable parameters, gamma and beta,
    initialized as vectors of 1 and 0 respectively

    you should use an epsilon of 1e-8

    Returns: a tensor of the activated output for the layer
    """
    heetal = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layers = tf.layers.Dense(units=n, activation=None,
                             kernel_initializer=heetal)
    lyr = layers(prev)
    mean, varz = tf.nn.moments(lyr, axes=[0])

    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)

    znorm = tf.nn.batch_normalization(lyr, mean=mean, variance=varz,
                                      offset=beta, scale=gamma,
                                      variance_epsilon=1e-8)
    return activation(znorm)
