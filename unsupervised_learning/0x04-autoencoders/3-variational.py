#!/usr/bin/env python3
"""
Autoencoders module
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
    representation
    Returns: encoder, decoder, auto
        encoder is the encoder model, which should output the latent
        representation, the mean, and the log variance, respectively
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and
    binary cross-entropy loss
    All layers should use a relu activation except for the mean and log
    variance layers in the encoder, which should use None, and the last layer
    in the decoder, which should use sigmoid
    """
    input_encoder = keras.layers.Input(shape=(input_dims,))
    input_encoded = input_encoder
    for n in hidden_layers:
        encoded = keras.layers.Dense(n, activation='relu')(input_encoded)
        input_encoded = encoded
    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.models.Model(input_encoder, latent)

    input_decoder = keras.layers.Input(shape=(latent_dims,))
    input_decoded = input_decoder
    for i, n in enumerate(hidden_layers[::-1]):
        activation = 'relu'
        decoded = keras.layers.Dense(n, activation=activation)(input_decoded)
        input_decoded = decoded
    decoded = keras.layers.Dense(
        input_dims, activation='sigmoid')(input_decoded)
    decoder = keras.models.Model(input_decoder, decoded)

    input_auto = keras.layers.Input(shape=(input_dims,))
    output_encoder = encoder(input_auto)
    output_decoder = decoder(output_encoder)

    auto = keras.models.Model(input_auto, output_decoder)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
