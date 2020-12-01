#!/usr/bin/env python3
"""
Autoencoders module
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
    representation
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and
    binary cross-entropy loss
    All layers should use a relu activation except for the last layer in the
    decoder, which should use sigmoid

    """
    input_encoder = keras.Input(shape=(input_dims,))
    encoder = keras.layers.Dense(hidden_layers[0],
                                 activation='relu')(input_encoder)
    for layer in hidden_layers[1:]:
        encoder = keras.layers.Dense(layer, activation='relu')(encoder)
    encoder = keras.layers.Dense(latent_dims, activation='relu')(encoder)
    encoder = keras.Model(inputs=input_encoder, outputs=encoder)

    input_decoder_input = keras.Input(shape=(latent_dims,))
    decoder = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(input_decoder_input)
    for layer in reversed(hidden_layers[:-1]):
        decoder = keras.layers.Dense(layer, activation='relu')(decoder)
    decoder = keras.layers.Dense(input_dims, activation='sigmoid')(decoder)
    decoder = keras.Model(inputs=input_decoder_input, outputs=decoder)

    output_encoder = encoder(input_encoder)
    output_decoder = decoder(output_encoder)
    auto = keras.Model(inputs=input_encoder, outputs=output_decoder)

    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
