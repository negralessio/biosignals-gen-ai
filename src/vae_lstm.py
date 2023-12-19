import tensorflow as tf

from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape, Dropout

from src.vae_base import BaseVAE, Sampling


class LSTMVAE(BaseVAE):
    """ Implementation of the BaseVAE using LSTM layers """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_encoder(self):
        """ Defines and returns Encoder architecture """

        inputs = Input(shape=(self.sequence_length, self.num_features))
        z = LSTM(64)(inputs)
        z_mean = Dense(self.latent_dims, name="z_mean")(z)
        z_log_var = Dense(self.latent_dims, name="z_log_var")(z)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(inputs=[inputs], outputs=[z_mean, z_log_var, z], name="encoder")

        return encoder

    def get_decoder(self):
        """ Defines and returns Decoder architecture """

        decoder_inputs = Input(shape=(self.latent_dims,))
        x = Dense(self.sequence_length * 1, activation="relu", name='Decode_1')(decoder_inputs)
        x = Reshape((self.sequence_length, 1), name='Decode_2')(x)
        x = LSTM(50, return_sequences=True)(x)
        decoder_output = TimeDistributed(Dense(self.num_features, activation='linear'), name='Decoder_Output_Layer')(x)
        decoder = tf.keras.Model(inputs=decoder_inputs, outputs=decoder_output, name="decoder")

        return decoder


