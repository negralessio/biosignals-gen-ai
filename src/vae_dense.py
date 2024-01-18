import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Dropout, LeakyReLU, BatchNormalization

from src.vae_base import BaseVAE, Sampling


class DenseVAE(BaseVAE):
    """ Implementation of the BaseVAE using Dense layers """

    def __init__(self, dropout_rate: float = 0.2, activation: str = "relu", *args, **kwargs):
        self.dropout_rate = dropout_rate
        self.activation = activation
        super().__init__(*args, **kwargs)

    def get_encoder(self):
        """ Defines and returns Encoder architecture """

        inputs = Input(shape=(self.sequence_length, self.num_features))
        z = Flatten()(inputs)
        z = Dense(32, activation=self.activation)(z)
        z = BatchNormalization()(z)
        z = Dense(32, activation=self.activation)(z)
        z = BatchNormalization()(z)
        z = Dense(16, activation=self.activation)(z)
        z_mean = Dense(self.latent_dims, name="z_mean")(z)
        z_log_var = Dense(self.latent_dims, name="z_log_var")(z)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(inputs=[inputs], outputs=[z_mean, z_log_var, z], name="encoder")

        return encoder

    def get_decoder(self):
        """ Defines and returns Decoder architecture """

        decoder_inputs = Input(shape=(self.latent_dims))
        x = Dense(16, activation=self.activation)(decoder_inputs)
        x = BatchNormalization()(x)
        x = Dense(32, activation=self.activation)(x)
        x = BatchNormalization()(x)
        x = Dense(32, activation=self.activation)(x)
        x = BatchNormalization()(x)
        x = Dense(self.sequence_length * self.num_features, name='decoder_final_dense')(x)
        decoder_output = Reshape(target_shape=(self.sequence_length, self.num_features))(x)
        decoder = tf.keras.Model(inputs=decoder_inputs, outputs=decoder_output, name="decoder")

        return decoder


class ImprovedDenseVAE(BaseVAE):
    def __init__(self, dropout_rate: float = 0.2, l2_reg_weight: float = 1e-5, *args, **kwargs):
        self.dropout_rate = dropout_rate
        self.l2_reg_weight = l2_reg_weight
        super().__init__(*args, **kwargs)

    def get_encoder(self):
        inputs = Input(shape=(self.sequence_length, self.num_features))
        z = Flatten()(inputs)
        z = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight))(z)
        z = BatchNormalization()(z)
        z = LeakyReLU()(z)
        z = Dropout(self.dropout_rate)(z)
        z = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight))(z)
        z = BatchNormalization()(z)
        z = LeakyReLU()(z)
        z = Dropout(self.dropout_rate)(z)
        z = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight))(z)
        z = BatchNormalization()(z)
        z = LeakyReLU()(z)
        z = Dropout(self.dropout_rate)(z)
        z = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight))(z)
        z = BatchNormalization()(z)
        z = LeakyReLU()(z)
        z = Dropout(self.dropout_rate)(z)
        z = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight))(z)
        z = LeakyReLU()(z)
        z_mean = Dense(self.latent_dims, name="z_mean")(z)
        z_log_var = Dense(self.latent_dims, name="z_log_var")(z)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(inputs=[inputs], outputs=[z_mean, z_log_var, z], name="encoder")
        return encoder

    def get_decoder(self):
        decoder_inputs = Input(shape=(self.latent_dims))
        x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight))(decoder_inputs)
        x = LeakyReLU()(x)
        x = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dense(self.sequence_length * self.num_features, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_weight), name='decoder_final_dense')(x)
        decoder_output = Reshape(target_shape=(self.sequence_length, self.num_features))(x)
        decoder = tf.keras.Model(inputs=decoder_inputs, outputs=decoder_output, name="decoder")
        return decoder