#!/usr/bin/env python3
"""
Keras Module
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent

    network is the model to train
    data is a numpy.ndarray of shape (m, nx) containing the input data
    labels is a one-hot numpy.ndarray of shape (m, classes) containing the
    labels of data
    batch_size is the size of the batch used for mini-batch gradient descent
    epochs is the number of passes through data for minibatch gradient descent
    verbose is a boolean that determines if output should be printed during
    training
    shuffle is a boolean that determines whether to shuffle the batches every
    epoch. Normally, it is a good idea to shuffle, but for reproducibility, we
    have chosen to set the default to False.
    validation_data is the data to validate the model with, if not None
    early_stopping is a bool that indicates whether early-stop should be used
        early stopping should only be performed if validation_data exists
        early stopping should be based on validation loss
    patience is the patience used for early stopping

    Returns: the History object generated after training the model
    """
    stop_call = None
    if validation_data:
        stop_call = [K.callbacks.EarlyStopping(patience=patience)]
    return network.fit(x=data, y=labels, batch_size=batch_size,
                       callbacks=stop_call, epochs=epochs, shuffle=shuffle,
                       validation_data=validation_data, verbose=verbose)
