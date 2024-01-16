# TODO: Saving Modell, Plotting and Saving History / Loss Curves, ...
import logging

import tensorflow as tf
import numpy as np
import src.utils as utils

from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Reshape, Dropout

from abc import ABC, abstractmethod

utils.setup_logging()
logger = logging.getLogger(__name__)


class Sampling(tf.keras.layers.Layer):
    """ Sampling layer class to sample from mean and log variance """

    def call(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class BaseVAE(tf.keras.Model, ABC):
    """ Abstract class defining the Variational Auto Encoder """
    def __init__(self, tensor: np.array, latent_dims: int, reconstruction_weight: int = 1, **kwargs):
        """

        :param tensor: np.array -- 3D tensor / input data
        :param latent_dims: int -- Number of latent dimensions
        :param reconstruction_weight: int -- How much more weight to add to the RL loss
        :param kwargs: dict
        """
        super().__init__(**kwargs)
        self.tensor = tensor
        self.latent_dims = latent_dims
        self.reconstruction_weight = reconstruction_weight

        self.sequence_length = tensor.shape[1]
        self.num_features = tensor.shape[2]

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @abstractmethod
    def get_encoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_decoder(self, **kwargs):
        raise NotImplementedError

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mean_squared_error(data, reconstruction),
                    axis=(1,)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = self.reconstruction_weight * reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.mean_squared_error(data, reconstruction),
                axis=(1,)
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))

        total_loss = self.reconstruction_weight * reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstructed = self.decoder(z)
        return reconstructed

