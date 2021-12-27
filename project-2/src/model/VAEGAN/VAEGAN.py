import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from Encoder import Encoder
from Generator import Generator
from Discriminator import Discriminator

class VAEGAN(keras.Model):

    def __init__(self, latent_dim, input_dim, output_dim, n_classes, train=True):
        super(VAEGAN, self).__init__()
        self.encoder = Encoder(latent_dim, input_dim)
        self.generator = Generator(output_dim)
        self.discriminator = Discriminator(n_classes)
        if train:
            self.enc_optimizer = keras.optimizers.Adam()
            self.gen_optimizer = keras.optimizers.Adam()
            self.disc_optimizer = keras.optimizers.Adam()
    
    def sample(self, z_mean, z_log_var):
        # sample using reparameterization trick
        eps = tf.random.normal(z_mean.shape)
        return z_mean + tf.exp(z_log_var*0.5)*eps
   
    def kl_loss(self, z_mean, z_log_var):
        return -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
 
    def generator_loss(self, y_true, y_pred):
        return - keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
   
    def discriminator_loss(self, y_true, y_pred):
        return keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred) 

    def discriminator_reconstruction_loss(self, h_real, h_fake):
        return tf.reduce_mean(tf.reduce_mean(tf.math.square(h_real-h_fake), axis=0))

    def vaegan_loss(self, x_real_580, x_real, y_real):
        z_mean, z_log_var = self.encoder(x_real_580)
        z_sampled = self.sample(z_mean, z_log_var)
        x_fake = self.generator(z_sampled)
        y_fake = tf.fill((x_fake.shape[0],), 2) 
        y_real_pred, h_real = self.discriminator(x_real)
        y_fake_pred, h_fake = self.discriminator(x_fake)
        
        # calculate losses
        kl_loss = self.kl_loss(z_mean, z_log_var)
        gen_fake_loss = self.generator_loss(y_fake, y_fake_pred)
        disc_real_loss = self.discriminator_loss(y_real, y_real_pred)
        disc_fake_loss = self.discriminator_loss(y_fake, y_fake_pred)
        disc_recon_loss = self.discriminator_reconstruction_loss(h_real, h_fake)

        enc_loss = kl_loss + disc_recon_loss
        gen_loss = gen_fake_loss + disc_recon_loss
        disc_loss = disc_real_loss + disc_fake_loss
        
        return enc_loss, gen_loss, disc_loss, x_fake, y_fake

    @tf.function
    def train_step(self, x_real_580, x_real, y_real):
        with tf.GradientTape() as enc_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            enc_loss, gen_loss, disc_loss, x_fake, y_fake = self.vaegan_loss(x_real_580, x_real, y_real)

        # calculate gradients
        enc_grads = enc_tape.gradient(enc_loss, self.encoder.trainable_variables)
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # apply gradients
        self.enc_optimizer.apply_gradients(zip(enc_grads, self.encoder.trainable_weights))
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_weights))
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_weights)) 

        return enc_loss, gen_loss, disc_loss, x_fake, y_fake

if __name__ == "__main__":
    obj = VAEGAN(20, 580, 2048,3)
