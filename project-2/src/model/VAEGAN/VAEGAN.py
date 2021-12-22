import tensorflow as tf
from tensorflow import keras
from keras import layers

from Encoder import Encoder
from Generator import Generator
from Discriminator import Discriminator

class VAEGAN(keras.Model):

    def __init__(self, latent_dim, n_inputs, n_classes):
        super(VAEGAN, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.generator = Generator(n_inputs)
        self.discriminator = Discriminator(n_classes)
    
    def sample(self, z_mean, z_log_var):
        # sample using reparameterization trick
        eps = tf.random.normal(z_mean.shape)
        return z_mean + tf.exp(z_log_var*0.5)*eps
   
    def kl_loss(self, z_mean, z_log_var):
        return -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
 
    def generator_loss(self, y_true, y_pred):
        return keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
   
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
 
    def interpolate(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z = [z1+(z2-z2)*t for t in np.linspace(0,1,10)]
        out = self.generator(z)
        return out

if __name__ == "__main__":
    self = VAEGAN(5,2,3)
