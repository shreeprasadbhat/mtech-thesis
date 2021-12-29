import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from Encoder import Encoder
from Generator import Generator
from Discriminator import Discriminator

class VAEGAN(keras.Model):

    def __init__(self, latent_dim, input_dim, output_dim, n_classes, batch_size, train=True):
        super(VAEGAN, self).__init__()
        self.encoder = Encoder(latent_dim, input_dim)
        self.generator = Generator(output_dim)
        self.discriminator = Discriminator(n_classes)
        if train:
            self.enc_optimizer = keras.optimizers.Adam()
            self.gen_optimizer = keras.optimizers.Adam()
            self.disc_optimizer = keras.optimizers.Adam()
    
    def sample(self, shape, z_mean=0., z_log_var=0.):
        # sample using reparameterization trick
        eps = tf.random.normal(shape)
        return z_mean + tf.exp(z_log_var*0.5)*eps
   
    def kl_loss(self, z_mean, z_log_var):
        return -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
     
    #def generator_loss(self, y_true, y_pred):
    #    return - keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
   
    def supervised_loss(self, y_true, y_pred):
        return keras.losses.SparseCategoricalCrossentropy() (y_true, y_pred) 
    
    def unsupervised_loss(self, y_true, y_pred):
        return keras.losses.BinaryCrossentropy() (y_true, y_pred)    

    def custom_activation(self, out):
        logexpsum = tf.reduce_sum(tf.exp(out), axis=-1, keepdims=True) 
        result = logexpsum / (logexpsum + 1.)
        return result

    def discriminator_reconstruction_loss(self, h_real, h_fake):
        return tf.reduce_mean(tf.reduce_mean(tf.math.square(h_real-h_fake), axis=0))

    def vaegan_loss(self, x_real_580, x_real, y_real):
        z_mean, z_log_var = self.encoder(x_real_580)
        z_sampled = self.sample(z_mean.shape, z_mean, z_log_var)
        z_prior = self.sample(z_mean.shape)

        x_fake = self.generator(z_sampled)
        x_enc = self.generator(z_prior)
        y_fake = tf.fill((x_fake.shape[0],), 2) 
        y_real_pred, h_real = self.discriminator(x_real)
        y_fake_pred, h_fake = self.discriminator(x_fake)
        _, h_enc = self.discriminator(x_enc)
        
        # calculate losses
        kl_loss = self.kl_loss(z_mean, z_log_var)
        #gen_fake_loss = self.generator_loss(y_fake, y_fake_pred)
        generator_loss = self.unsupervised_loss(tf.ones_like(y_fake), self.custom_activation(h_fake))
        supervised_loss = self.supervised_loss(y_real, y_real_pred)
        unsupervised_loss = self.unsupervised_loss(tf.ones_like(y_real), self.custom_activation(h_real))
        unsupervised_loss += self.unsupervised_loss(tf.zeros_like(y_fake), self.custom_activation(h_fake))
        disc_recon_loss = self.discriminator_reconstruction_loss(h_real, h_enc)

        enc_loss = kl_loss + disc_recon_loss
        gen_loss = disc_recon_loss + generator_loss
        disc_loss = supervised_loss + unsupervised_loss
        
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
