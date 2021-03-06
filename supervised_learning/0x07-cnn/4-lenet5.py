#!/usr/bin/env python3
"""
CNN Module
"""
import tensorflow as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow

    x is a tf.placeholder of shape (m, 28, 28, 1) containing the input images
    for the network
        m is the number of images

    y is a tf.placeholder of shape (m, 10) containing the one-hot labels for
    the network

    The model should consist of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes

    All layers requiring initialization should initialize their kernels with
    the he_normal initialization method:
        tf.contrib.layers.variance_scaling_initializer()

    All hidden layers requiring activation should use the relu activation
    function

    Returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
            (with default hyperparameters)
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network

    """
    init = tf.contrib.layers.variance_scaling_initializer()
    cvv1 = tf.layers.Conv2D(filters=6,
                            kernel_size=5,
                            padding="same",
                            activation=tf.nn.relu,
                            kernel_initializer=init)(x)
    pooll1 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(cvv1)

    cvv2 = tf.layers.Conv2D(filters=16,
                            kernel_size=5,
                            padding="valid",
                            activation=tf.nn.relu,
                            kernel_initializer=init)(pooll1)
    pooll2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(cvv2)
    pooll2 = tf.layers.Flatten()(pooll2)

    full1 = tf.layers.Dense(units=120,
                            activation=tf.nn.relu,
                            kernel_initializer=init)(pooll2)

    full2 = tf.layers.Dense(units=84,
                            activation=tf.nn.relu,
                            kernel_initializer=init)(full1)

    output = tf.layers.Dense(units=10, kernel_initializer=init)(full2)

    smax = tf.nn.softmax(output)
    loss = tf.losses.softmax_cross_entropy(y, output)
    adam = tf.train.AdamOptimizer().minimize(loss)
    pred = tf.argmax(y, 1)
    tagg = tf.argmax(output, 1)
    eqty = tf.equal(pred, tagg)
    accr = tf.reduce_mean(tf.cast(eqty, tf.float32))

    return smax, adam, loss, accr
