import tensorflow as tf
from tensorflow import keras
from keras import layers 

from Generator import Generator
from Discriminator import Discriminator

class GAN(keras.Model):
    def __init__(self, input_dim, latent_dim, n_classes ):
        super().__init__()
        self.latent_dim = latent_dim

        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator(input_dim, n_classes) 
                      
    def supervised_loss(self, y_true, y_pred):
        return keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)

    def unsupervised_loss(self, y_true, y_pred):
        return keras.losses.BinaryCrossentropy() (y_true, y_pred)

    def custom_activation(self, out):
        logexpsum = tf.reduce_sum(tf.exp(out), axis=-1, keepdims=True)
        result = logexpsum / ( logexpsum + 1.)
        return result
 
    def gan_loss(self, x_real, y_real, batch_size):
        z = tf.random.normal((batch_size, self.latent_dim))
        x_fake = self.generator(z) 
        y_fake_pred, h_fake = self.discriminator(x_fake)
        y_real_pred, h_real = self.discriminator(x_real)

        gen_loss = self.unsupervised_loss(tf.ones((batch_size,)), self.custom_activation(h_fake))
        supervised_loss = self.supervised_loss(y_real, y_real_pred)
        unsupervised_loss = self.unsupervised_loss(tf.ones((batch_size,)), self.custom_activation(h_real))
        unsupervised_loss += self.unsupervised_loss(tf.zeros((batch_size,)), self.custom_activation(h_fake))

        disc_loss = supervised_loss + unsupervised_loss

        return gen_loss, disc_loss

    def def_optimizer(self, lr_gen=1e-3, lr_disc=1e-3):
        self.gen_optimizer = keras.optimizers.Adam(lr_gen)
        self.disc_optimizer = keras.optimizers.Adam(lr_disc)

    @tf.function
    def train_step(self, x_real, y_real):
        
        # calculate the loss
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
           gen_loss, disc_loss = self.gan_loss(x_real, y_real, x_real.shape[0])  
        
        # calculate the gradients        
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # apply gradients
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_weights))  
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_weights))
        
        return gen_loss, disc_loss  

if __name__ == "__main__":
    GAN(2048,16,2)
