#!/usr/bin/env python3
"""
Tensorflow Module
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network

    X is a numpy.ndarray containing the input data to evaluate
    Y is a numpy.ndarray containing the one-hot labels for X
    save_path is the location to load the model from

    Returns: the network’s prediction, accuracy, and loss, respectively
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        fd = {x: X, y: Y}
        y_pred, accy, loss = sess.run([y_pred, accy, loss], feed_dict=fd)
    return y_pred, accy, loss
