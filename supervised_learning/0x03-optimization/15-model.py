#!/usr/bin/env python3
"""
Optimization Module
"""
import numpy as np
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
    if activation is None:
        return znorm
    return activation(znorm)


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network

    x is the placeholder for the input data
    layer_sizes is a list with the number of nodes in each layer of the network
    activations is a list with the act. functions for each layer of the network

    Returns: the prediction of the network in tensor form
    """
    prd = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        prd = create_batch_norm_layer(x, layer_sizes[i], activations[i])
    return prd


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


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction

    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions

    Returns: a tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation in tensorflow using inverse time
    decay

    alpha is the original learning rate
    decay_rate is the weight used to determine the rate at which alpha'll decay
    global_step is the number of passes of gradient descent that have elapsed
    decay_step is the number of passes of gradient descent that should occur
    before alpha is decayed further
    the learning rate decay should occur in a stepwise fashion

    Returns: the learning rate decay operation
    """
    return (tf.train.inverse_time_decay(alpha,
            global_step, decay_step, decay_rate, staircase=True))


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm

    loss is the loss of the network
    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero

    Returns: the Adam optimization operation
    """
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way

    X is the first numpy.ndarray of shape (m, nx) to shuffle
        m is the number of data points
        nx is the number of features in X
    Y is the second numpy.ndarray of shape (m, ny) to shuffle
        m is the same number of data points as in X
        ny is the number of features in Y

    Returns: the shuffled X and Y matrices
    """
    idsf = np.random.permutation(X.shape[0])
    return X[idsf], Y[idsf]


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow using Adam
    optimization, mini-batch gradient descent, learning rate decay, and batch
    normalization

    Data_train is a tuple containing the training inputs and training labels

    Data_valid is a tple containing the validation inputs and validation labels

    layers is a list containing the n of nodes in each layer of the network

    activation is a list containing the activation functions used for each
    layer of the network alpha is the learning rate

    beta1 is the weight for the first moment of Adam Optimization
    beta2 is the weight for the second moment of Adam Optimization
    epsilon is a small number used to avoid division by zero

    decay_rate is the decay rate for inverse time decay of the learning rate
    (the corresponding decay step should be 1)

    batch_size is the number of data points that should be in a mini-batch
    epochs is the number of times the training should pass through the dataset
    save_path is the path where the model should be saved to

    Returns: the path where the model was saved
    """
    x_shape = (None, Data_train[0].shape[1])
    y_shape = (None, Data_train[1].shape[1])
    x = (tf.placeholder(tf.float32, shape=x_shape, name='x'))
    y = (tf.placeholder(tf.float32, shape=y_shape, name='y'))

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        mini_batch = Data_train[0].shape[0] / batch_size
        iterations = int(mini_batch)
        if mini_batch > iterations:
            iterations += 1

        feed_dict_trn = {x: Data_train[0], y: Data_train[1]}
        feed_dict_vld = {x: Data_valid[0], y: Data_valid[1]}

        for i in range(epochs + 1):
            trn_cost = sess.run(loss, feed_dict_trn)
            vld_cost = sess.run(loss, feed_dict_vld)
            trn_accy = sess.run(accuracy, feed_dict_trn)
            vld_accy = sess.run(accuracy, feed_dict_vld)

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(trn_cost))
            print("\tTraining Accuracy: {}".format(trn_accy))
            print("\tValidation Cost: {}".format(vld_cost))
            print("\tValidation Accuracy: {}".format(vld_accy))

            if i < epochs:
                X_sff, Y_sff = shuffle_data(Data_train[0], Data_train[1])
                for step in range(iterations):
                    start = step * batch_size
                    if step == iterations - 1 and (mini_batch > iterations):
                        end = int(
                            batch_size * (step + mini_batch - iterations + 1))
                    else:
                        end = batch_size * (1 + step)
                    feed_dict = {x: X_sff[start: end], y: Y_sff[start: end]}
                    sess.run(train_op, feed_dict)
                    if step != 0 and (step + 1) % 100 == 0:
                        print("\tStep {}:".format(step + 1))
                        tmb_cost = sess.run(loss, feed_dict)
                        print("\t\tCost: {}".format(tmb_cost))
                        tmb_accy = sess.run(accuracy, feed_dict)
                        print("\t\tAccuracy: {}".format(tmb_accy))
            sess.run(tf.assign(global_step, global_step + 1))
        save_path = saver.save(sess, save_path)
    return save_path
