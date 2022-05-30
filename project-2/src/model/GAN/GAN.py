import tensorflow as tf
from tensorflow import keras
from keras import layers 
import matplotlib.pyplot as plt
import numpy as np

class GAN(keras.Model):
    def __init__(self, latent_dim, generator, discriminator, z, scaler=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator
        self.z = z # redshift, required in GANMonitor
        self.scaler = scaler

    def compile(self, 
            d_optimizer, 
            g_optimizer,
            binary_cross_entropy_loss_fn):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.binary_cross_entropy_loss_fn = binary_cross_entropy_loss_fn    
        self.d_acc = tf.keras.metrics.BinaryAccuracy(name='d_acc') 
        self.g_loss = tf.keras.metrics.Mean(name='g_loss')
        self.d_loss = tf.keras.metrics.Mean(name='d_loss')


    @tf.function
    def train_step(self, data):

        x_real = data

        batch_size = tf.shape(x_real)[0]

        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        x_fake = self.generator(z)

        x = tf.concat([x_real, x_fake], axis=0)
        y = tf.concat([tf.ones(shape=(batch_size)), tf.zeros(shape=(batch_size))], axis=0)
        y_noisy = y + 0.05 * tf.random.uniform(tf.shape(y))
        with tf.GradientTape() as tape:
            y_pred = self.discriminator(x)
            d_loss = self.binary_cross_entropy_loss_fn(y_noisy, y_pred)
        
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        self.d_loss.update_state(d_loss)
        self.d_acc.update_state(y, y_pred) 

        with tf.GradientTape() as tape:
            y_pred = self.discriminator(self.generator(z))
            y_true = tf.ones(shape=(batch_size, 1))
            g_loss = self.binary_cross_entropy_loss_fn(y_true, y_pred)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        self.g_loss.update_state(g_loss)
         
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):

        x_real = data

        batch_size = tf.shape(x_real)[0]
        
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        x_fake = self.generator(z)
        x = tf.concat([x_real, x_fake], axis=0)
        y = tf.concat([tf.ones(shape=(batch_size)), tf.zeros(shape=(batch_size))], axis=0)

        y_pred = self.discriminator(x)
        d_loss = self.binary_cross_entropy_loss_fn(y, y_pred)
        
        self.d_loss.update_state(d_loss)
        self.d_acc.update_state(y, y_pred) 

        y_pred = self.discriminator(x_fake)
        y_true = tf.ones(shape=(batch_size, 1))
        g_loss = self.binary_cross_entropy_loss_fn(y_true, y_pred)

        self.g_loss.update_state(g_loss)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.d_loss, self.g_loss, self.d_acc]

if __name__ == "__main__":
    from Generator import Generator
    from Discriminator import Discriminator

    discriminator = Discriminator(2048)
    gan = GAN(16, Generator(2), Discriminator(3))
