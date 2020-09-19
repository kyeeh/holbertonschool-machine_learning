#!/usr/bin/env python3
"""
Deep CNN Module
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in Deep Residual Learning
    for Image Recognition (2015)

    You can assume the input data will have shape (224, 224, 3)

    All convolutions inside and outside the blocks should be followed by batch
    normalization along the channels axis and a rectified linear activation
    (ReLU), respectively.

    All weights should use he normal initialization

    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)

    cvv1 = K.layers.Conv2D(filters=64,
                           kernel_size=7,
                           padding='same',
                           strides=2,
                           kernel_initializer=init)(X)

    cvv1 = K.layers.BatchNormalization(axis=3)(cvv1)
    cvv1 = K.layers.Activation('relu')(cvv1)

    cvb1 = K.layers.MaxPool2D(pool_size=3,
                              strides=2,
                              padding='same')(cvv1)

    cvb1 = projection_block(cvb1, [64, 64, 256], 1)
    for i in range(2):
        cvb1 = identity_block(cvb1, [64, 64, 256])

    cvb2 = projection_block(cvb1, [128, 128, 512])
    for i in range(3):
        cvb2 = identity_block(cvb2, [128, 128, 512])

    cvb3 = projection_block(cvb2, [256, 256, 1024])
    for i in range(5):
        cvb3 = identity_block(cvb3, [256, 256, 1024])

    cvb4 = projection_block(cvb3, [512, 512, 2048])
    for i in range(2):
        cvb4 = identity_block(cvb4, [512, 512, 2048])

    apool5 = K.layers.AveragePooling2D(padding='same', pool_size=7)(cvb4)

    output = K.layers.Dense(1000, activation='softmax')(apool5)
    model = K.models.Model(inputs=X, outputs=output)
    return model
