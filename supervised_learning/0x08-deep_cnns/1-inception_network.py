#!/usr/bin/env python3
"""
Deep CNN Module
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds an inception network as described in Going Deeper with
    Convolutions (2014)

    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the inception block should use a
    rectified linear activation (ReLU)

    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)

    cvv1 = K.layers.Conv2D(filters=64,
                           kernel_size=7,
                           padding='same',
                           strides=2,
                           activation='relu',
                           kernel_initializer=init)(X)

    mpool1 = K.layers.MaxPool2D(padding='same',
                                pool_size=3,
                                strides=2)(cvv1)

    cvv2 = K.layers.Conv2D(filters=64,
                           kernel_size=1,
                           padding='same',
                           strides=1,
                           activation='relu',
                           kernel_initializer=init)(mpool1)

    cvv3 = K.layers.Conv2D(filters=192,
                           kernel_size=3,
                           padding='same',
                           strides=1,
                           activation='relu',
                           kernel_initializer=init)(cvv2)

    mpool2 = K.layers.MaxPool2D(padding='same',
                                pool_size=3,
                                strides=2)(cvv3)

    incp3a = inception_block(mpool2, [64, 96, 128, 16, 32, 32])
    incp3b = inception_block(incp3a, [128, 128, 192, 32, 96, 64])

    mpool3 = K.layers.MaxPool2D(padding='same',
                                pool_size=3,
                                strides=2)(incp3b)

    incp4a = inception_block(mpool3, [192, 96, 208, 16, 48, 64])
    incp4b = inception_block(incp4a, [160, 112, 224, 24, 64, 64])
    incp4c = inception_block(incp4b, [128, 128, 256, 24, 64, 64])
    incp4d = inception_block(incp4c, [112, 144, 288, 32, 64, 64])
    incp4e = inception_block(incp4d, [256, 160, 320, 32, 128, 128])

    mpool4 = K.layers.MaxPool2D(padding='same',
                                pool_size=3,
                                strides=2)(incp4e)

    incp5a = inception_block(mpool4, [256, 160, 320, 32, 128, 128])
    incp5b = inception_block(incp5a, [384, 192, 384, 48, 128, 128])

    apool5 = K.layers.AveragePooling2D(padding='same', pool_size=7)(incp5b)

    dropout = K.layers.Dropout(rate=0.4)(apool5)
    output = K.layers.Dense(1000, activation='softmax')(dropout)

    model = K.models.Model(inputs=X, outputs=output)
    return model
