import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import matplotlib.pyplot as plt

class Sampling(layers.Layer):
    """Uses (z_mean, z_logvar) to sample z, the vector encoding a digit."""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim 

    def call(self, inputs):
        z_mean, z_log_sigma = inputs
        epsilon = tf.random.normal(shape=(self.latent_dim,))
        return z_mean + tf.math.exp(z_log_sigma) * epsilon


class VAEGAN(keras.Model):
    def __init__(self, latent_dim, encoder, generator, discriminator, s_discriminator, u_discriminator, z=None, scaler=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator
        self.s_discriminator = s_discriminator
        self.u_discriminator = u_discriminator
        self.sampling = Sampling(latent_dim)
        self.z = z
        self.scaler = scaler

    def compile(self, 
            s_optimizer, 
            u_optimizer, 
            g_optimizer,
            e_optimizer,
            binary_cross_entropy_loss_fn,
            sparse_categorical_cross_entropy_loss_fn):
        super().compile()
        self.g_optimizer = g_optimizer
        self.s_optimizer = s_optimizer
        self.u_optimizer = u_optimizer
        self.e_optimizer = e_optimizer
        self.binary_cross_entropy_loss_fn = binary_cross_entropy_loss_fn    
        self.sparse_categorical_cross_entropy_loss_fn = sparse_categorical_cross_entropy_loss_fn
        self.mse = tf.keras.losses.MeanSquaredError()
        self.s_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='s_acc') 
        self.u_acc = tf.keras.metrics.BinaryAccuracy(name='u_acc') 
        self.e_loss = tf.keras.metrics.Mean(name='e_loss')
        self.g_loss = tf.keras.metrics.Mean(name='g_loss')
        self.s_loss = tf.keras.metrics.Mean(name='s_loss')
        self.u_loss = tf.keras.metrics.Mean(name='u_loss')

    # Note: You could also analytically compute the KL term, but here you 
    # incorporate all three terms in the Monte Carlo estimator for simplicity.
    def kl_loss(self, z_mean, z_logvar, raxis=1):
        return -0.5 * tf.reduce_mean(
            z_logvar - tf.square(z_mean) - tf.exp(z_logvar) + 1
        )

    @tf.function
    def train_step(self, data):

        x_real_580, x_real, y_real = data

        batch_size = tf.shape(x_real)[0]

        with tf.GradientTape() as tape:
            y_pred = self.s_discriminator(x_real)
            print(y_real.shape, y_pred.shape)
            s_loss = self.sparse_categorical_cross_entropy_loss_fn(y_real, y_pred)

        grads = tape.gradient(s_loss, self.s_discriminator.trainable_weights)
        self.s_optimizer.apply_gradients(zip(grads, self.s_discriminator.trainable_weights))
        self.s_loss.update_state(s_loss)
        self.s_acc.update_state(y_real, y_pred)
        
        z_prior = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        x = tf.concat([x_real, self.generator(z_prior)], axis=0)
        y = tf.concat([tf.ones(shape=(batch_size)), tf.zeros(shape=(batch_size))], axis=0)
        y_noisy = y + 0.05 * tf.random.uniform(tf.shape(y))
        with tf.GradientTape() as tape:
            y_pred = self.u_discriminator(x)
            u_loss = self.binary_cross_entropy_loss_fn(y_noisy, y_pred)
        
        grads = tape.gradient(u_loss, self.u_discriminator.trainable_weights)
        self.u_optimizer.apply_gradients(zip(grads, self.u_discriminator.trainable_weights))
        self.u_loss.update_state(u_loss)
        self.u_acc.update_state(y, y_pred) 
        
        with tf.GradientTape() as tape:

            y_pred = self.u_discriminator(self.generator(z_prior))
            y_true = tf.ones(shape=(batch_size, 1))
            g_loss = self.binary_cross_entropy_loss_fn(y_true, y_pred)

            z_mean, z_logvar = self.encoder(x_real_580)
            z = self.sampling((z_mean, z_logvar)) # reparameterize
    
            y_pred2 = self.discriminator(self.generator(z))
            y_true2 = self.discriminator(x_real)
            g_loss += self.binary_cross_entropy_loss_fn(y_true2, y_pred2)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        self.g_loss.update_state(g_loss)

        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder(x_real_580)
            z = self.sampling((z_mean, z_logvar)) # reparameterize
            #x_pred = self.discriminator(self.generator(z))
            #x_true = self.discriminator(x_real)
            x_pred = self.generator(z)
            x_true = x_real
            recon_loss = self.mse(x_pred, x_true)
            kl_loss = self.kl_loss(z_mean, z_logvar)
            e_loss = recon_loss + kl_loss

        grads = tape.gradient(e_loss, self.encoder.trainable_weights)
        self.e_optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights))
        self.e_loss.update_state(e_loss) 

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):

        x_real_580, x_real, y_real = data

        batch_size = tf.shape(x_real)[0]
        
        y_pred = self.s_discriminator(x_real)
        s_loss = self.sparse_categorical_cross_entropy_loss_fn(y_real, y_pred)
        
        self.s_loss.update_state(s_loss)
        self.s_acc.update_state(y_real, y_pred)
        
        z_prior = tf.random.normal(shape=(batch_size, self.latent_dim))

        x = tf.concat([x_real, self.generator(z_prior)], axis=0)
        y = tf.concat([tf.ones(shape=(batch_size)), tf.zeros(shape=(batch_size))], axis=0)
        y_noisy = y + 0.05 * tf.random.uniform(tf.shape(y))

        y_pred = self.u_discriminator(x)
        u_loss = self.binary_cross_entropy_loss_fn(y_noisy, y_pred)
        
        self.u_loss.update_state(u_loss)
        self.u_acc.update_state(y, y_pred) 

        y_pred = self.u_discriminator(self.generator(z_prior))
        y_true = tf.ones(shape=(batch_size, 1))
        g_loss = self.binary_cross_entropy_loss_fn(y_true, y_pred)

        z_mean, z_logvar = self.encoder(x_real_580)
        z = self.sampling((z_mean, z_logvar)) # reparameterize

        y_pred2 = self.discriminator(self.generator(z))
        y_true2 = self.discriminator(x_real)
        g_loss += self.binary_cross_entropy_loss_fn(y_true2, y_pred2)

        self.g_loss.update_state(g_loss)

        x_pred = self.discriminator(self.generator(z))
        x_true = self.discriminator(x_real)
        #x_pred = self.generator(z)
        #x_true = x_real
        recon_loss = self.mse(x_pred, x_true)
        kl_loss = self.kl_loss(z_mean, z_logvar)
        e_loss = recon_loss + kl_loss
        
        self.e_loss.update_state(e_loss) 

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
   
    def my_predict(self, x):

        z_mean, z_logvar = self.encoder(x) 
        z = self.sampling((z_mean, z_logvar)) # re-parameterize

        x_interpolate = self.generator(z)

        plt.scatter(self.z, x_interpolate.numpy(), s=4)
        plt.xlim(0,1.2)
        plt.ylim(0,1.2)
        plt.show()

        pred = self.s_discriminator(x_interpolate)
        print(pred)
        print('Class : ',tf.argmax(pred, axis=-1))



    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.e_loss, self.s_loss, self.u_loss, self.g_loss, self.s_acc, self.u_acc]

if __name__ == "__main__":
    from Generator import Generator
    from Discriminator import Discriminator
    from SupervisedDiscriminator import SupervisedDiscriminator
    from UnsupervisedDiscriminator import UnsupervisedDiscriminator
    discriminator = Discriminator(2048)
    sgan = SGAN(16, Generator(2), SupervisedDiscriminator(3, discriminator), UnsupervisedDiscriminator(discriminator))
