import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv1D, Conv1DTranspose

from src.vae_base import BaseVAE, Sampling


class ConvVAE(BaseVAE):
    """ Implementation of the BaseVAE using Convolutional layers """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder_last_dense_dim = None

    def get_encoder(self):
        """ Defines and returns Encoder architecture """

        inputs = Input(shape=(self.sequence_length, self.num_features))

        z = Conv1D(filters=5, kernel_size=3, strides=2, activation="relu", padding="same")(inputs)
        z = Conv1D(filters=25, kernel_size=3, strides=2, activation="relu", padding="same")(z)
        z = Conv1D(filters=50, kernel_size=3, strides=2, activation="relu", padding="same")(z)
        z = Conv1D(filters=100, kernel_size=3, strides=2, activation="relu", padding="same")(z)
        z = Flatten()(z)
        self.encoder_last_dense_dim = z.get_shape()[-1]

        z_mean = Dense(self.latent_dims, name="z_mean")(z)
        z_log_var = Dense(self.latent_dims, name="z_log_var")(z)
        z = Sampling()([z_mean, z_log_var])

        encoder = tf.keras.Model(inputs=[inputs], outputs=[z_mean, z_log_var, z], name="encoder")
        return encoder

    def get_decoder(self):
        """ Defines and returns Decoder architecture """

        decoder_inputs = Input(shape=(self.latent_dims))

        x = Dense(self.encoder_last_dense_dim, name="dec_dense", activation='relu')(decoder_inputs)
        x = Reshape(target_shape=(-1, 100), name="dec_reshape")(x)
        x = Conv1DTranspose(
            filters=50,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu')(x)
        x = Conv1DTranspose(
            filters=25,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu')(x)
        x = Conv1DTranspose(
            filters=5,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu')(x)
        x = Flatten(name='dec_flatten')(x)
        x = Dense(self.sequence_length * self.num_features, name="decoder_dense_final")(x)
        decoder_output = Reshape(target_shape=(self.sequence_length, self.num_features))(x)

        decoder = tf.keras.Model(inputs=decoder_inputs, outputs=decoder_output, name="decoder")
        return decoder
