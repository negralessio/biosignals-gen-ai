import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape

from src.vae_base import BaseVAE, Sampling


class DenseVAE(BaseVAE):
    """ Implementation of the BaseVAE using Dense layers """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_encoder(self):
        """ Defines and returns Encoder architecture """

        inputs = Input(shape=(self.sequence_length, self.num_features))
        z = Flatten()(inputs)
        z = Dense(256, activation="relu")(z)
        z = Dense(128, activation="relu")(z)
        z = Dense(64, activation="relu")(z)
        z = Dense(32, activation="relu")(z)
        z = Dense(16, activation="relu")(z)
        z_mean = Dense(self.latent_dims, name="z_mean")(z)
        z_log_var = Dense(self.latent_dims, name="z_log_var")(z)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(inputs=[inputs], outputs=[z_mean, z_log_var, z], name="encoder")

        return encoder

    def get_decoder(self):
        """ Defines and returns Decoder architecture """

        decoder_inputs = Input(shape=(self.latent_dims))
        x = Dense(16, activation="relu")(decoder_inputs)
        x = Dense(32, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        x = Dense(256, activation="relu")(x)
        x = Dense(self.sequence_length * self.num_features, name='decoder_final_dense')(x)
        decoder_output = Reshape(target_shape=(self.sequence_length, self.num_features))(x)
        decoder = tf.keras.Model(inputs=decoder_inputs, outputs=decoder_output, name="decoder")

        return decoder
