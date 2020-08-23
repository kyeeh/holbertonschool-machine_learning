#!/usr/bin/env python3
"""
Optimization Module
"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent

    X_train is a numpy.ndarray of shape (m, 784) containing the training data

        m is the number of data points
        784 is the number of input features

    Y_train is a one-hot numpy.ndarray of shape (m, 10) of training labels

        10 is the number of classes the model should classify

    X_valid is a np.ndarray of shape (m, 784) with the validation data
    Y_valid is a one-hot np.ndarray of shape (m, 10) with the validation labels
    batch_size is the number of data points in a batch
    epochs is the number of times the training should pass through the dataset
    load_path is the path from which to load the model
    save_path is the path to where the model should be saved after training

    Returns: the path where the model was saved
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        mini_batch = X_train.shape[0] / batch_size
        iterations = int(mini_batch)
        if mini_batch > iterations:
            iterations += 1

        for i in range(epochs + 1):
            trn_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            vld_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            trn_accy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            vld_accy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(trn_cost))
            print("\tTraining Accuracy: {}".format(trn_accy))
            print("\tValidation Cost: {}".format(vld_cost))
            print("\tValidation Accuracy: {}".format(vld_accy))

            if i < epochs:
                X_sff, Y_sff = shuffle_data(X_train, Y_train)
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
        save_path = saver.save(sess, save_path)
    return save_path
