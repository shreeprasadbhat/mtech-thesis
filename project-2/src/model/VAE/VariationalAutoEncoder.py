import sys
import os
import tensorflow as tf
from tensorflow.keras import layers

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.model.VAE.Encoder import Encoder
from src.model.VAE.Decoder import Decoder

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim 

    def call(self, inputs):
        z_mean, z_log_sigma = inputs
        epsilon = tf.random.normal(shape=(self.latent_dim,))
        return z_mean + tf.math.exp(z_log_sigma) * epsilon


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        input_dim,
        latent_dim=1,
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim)
        self.sampling = Sampling(latent_dim)

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling((z_mean, z_log_var))
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        #self.add_loss(kl_loss)
        return reconstructed

if __name__ == '__main__':
    vae = VariationalAutoEncoder(580, 16)
