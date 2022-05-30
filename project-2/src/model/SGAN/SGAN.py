import tensorflow as tf
from tensorflow import keras
from keras import layers 

class SGAN(keras.Model):
    def __init__(self, latent_dim, generator, s_discriminator, u_discriminator, z=None, scaler=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = generator
        self.s_discriminator = s_discriminator
        self.u_discriminator = u_discriminator
        self.z = z # redshift, required in GANMonitor
        self.scaler = scaler

    def compile(self, 
            s_optimizer, 
            u_optimizer, 
            g_optimizer,
            binary_cross_entropy_loss_fn,
            sparse_categorical_cross_entropy_loss_fn):
        super().compile()
        self.g_optimizer = g_optimizer
        self.s_optimizer = s_optimizer
        self.u_optimizer = u_optimizer
        self.binary_cross_entropy_loss_fn = binary_cross_entropy_loss_fn    
        self.sparse_categorical_cross_entropy_loss_fn = sparse_categorical_cross_entropy_loss_fn
        self.s_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='s_acc') 
        self.u_acc = tf.keras.metrics.BinaryAccuracy(name='u_acc') 
        self.g_loss = tf.keras.metrics.Mean(name='g_loss')
        self.s_loss = tf.keras.metrics.Mean(name='s_loss')
        self.u_loss = tf.keras.metrics.Mean(name='u_loss')


    @tf.function
    def train_step(self, data):
        x_real, y_real = data

        batch_size = tf.shape(x_real)[0]

        with tf.GradientTape() as tape:
            y_pred = self.s_discriminator(x_real)
            s_loss = self.sparse_categorical_cross_entropy_loss_fn(y_real, y_pred)

        grads = tape.gradient(s_loss, self.s_discriminator.trainable_weights)
        self.s_optimizer.apply_gradients(zip(grads, self.s_discriminator.trainable_weights))
        self.s_loss.update_state(s_loss)
        self.s_acc.update_state(y_real, y_pred)

        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        x = tf.concat([x_real, self.generator(z)], axis=0)
        y = tf.concat([tf.ones(shape=(batch_size)), tf.zeros(shape=(batch_size))], axis=0)
        y_noisy = y + 0.05 * tf.random.uniform(tf.shape(y))
        with tf.GradientTape() as tape:
            y_pred = self.u_discriminator(x)
            u_loss = self.binary_cross_entropy_loss_fn(y_noisy, y_pred)
        
        grads = tape.gradient(u_loss, self.u_discriminator.trainable_weights)
        self.u_optimizer.apply_gradients(zip(grads, self.u_discriminator.trainable_weights))
        self.u_loss.update_state(u_loss)
        self.u_acc.update_state(y, y_pred) 

        z = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            y_pred = self.u_discriminator(self.generator(z))
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
        x_real, y_real = data

        batch_size = tf.shape(x_real)[0]

        z = tf.random.normal(shape=(batch_size, self.latent_dim))

        y_pred = self.s_discriminator(x_real)
        s_loss = self.sparse_categorical_cross_entropy_loss_fn(y_real, y_pred)
        self.s_loss.update_state(s_loss)
        self.s_acc.update_state(y_real, y_pred)

        y_pred = self.u_discriminator(self.generator(z))
        y_true = tf.zeros(shape=(batch_size, 1))
        fake_loss = self.binary_cross_entropy_loss_fn(y_true, y_pred)
        self.u_loss.update_state(fake_loss)
        self.u_acc.update_state(y_true, y_pred)
        
        y_pred = self.u_discriminator(x_real)
        y_true = tf.ones(shape=(batch_size, 1))
        real_loss = self.binary_cross_entropy_loss_fn(y_true, y_pred)
        self.u_loss.update_state(y_true, y_pred)
        self.u_acc.update_state(y_true, y_pred)
        
        z = tf.random.normal(shape=(batch_size, self.latent_dim))

        y_pred = self.u_discriminator(self.generator(z))
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
        return [self.s_loss, self.u_loss, self.g_loss, self.s_acc, self.u_acc]

if __name__ == "__main__":
    from Generator import Generator
    from Discriminator import Discriminator
    from SupervisedDiscriminator import SupervisedDiscriminator
    from UnsupervisedDiscriminator import UnsupervisedDiscriminator
    discriminator = Discriminator(2048)
    sgan = SGAN(16, Generator(2), SupervisedDiscriminator(3, discriminator), UnsupervisedDiscriminator(discriminator))
