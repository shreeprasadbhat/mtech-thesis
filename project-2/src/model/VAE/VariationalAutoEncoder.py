import tensorflow as tf
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs, latent_dim=1):
        z_mean, z_log_sigma = inputs
        epsilon = tf.random.normal(shape=(latent_dim,))
        return z_mean + tf.math.exp(z_log_sigma) * epsilon

class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=1, intermediate_dim1=64, intermediate_dim2=32, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj1 = layers.Dense(intermediate_dim1, activation="relu")
        self.dense_proj2 = layers.Dense(intermediate_dim2, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj1(inputs)
        x = self.dense_proj2(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim1=32, intermediate_dim2=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj1 = layers.Dense(intermediate_dim1, activation="relu")
        self.dense_proj2 = layers.Dense(intermediate_dim2, activation="relu")
        self.dense_output = layers.Dense(original_dim, activation="tanh")

    def call(self, inputs):
        x = self.dense_proj1(inputs)
        x = self.dense_proj2(x)
        return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        original_dim,
        intermediate_dim1=32,
        intermediate_dim2=64,
        latent_dim=1,
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim1=intermediate_dim1, intermediate_dim2=intermediate_dim2)
        self.decoder = Decoder(original_dim, intermediate_dim1=intermediate_dim1, intermediate_dim2=intermediate_dim2)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        #self.add_loss(kl_loss)
        return reconstructed

if __name__ == '__main__':
    vae = VariationalAutoEncoder(2)
