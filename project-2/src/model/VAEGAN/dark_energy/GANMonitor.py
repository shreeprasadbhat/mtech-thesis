import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim=20):
        self.latent_dim = 20
        self.batch_size = 1 
        self.outdir = './runs/runs01/'

    def on_epoch_end(self, epoch, logs=None):

        if epoch % 1 == 0:

            z = tf.random.normal(shape=(1, self.latent_dim))
            #z_mean, z_logvar = self.modelencoder(x_test_580[0].reshape(-1,580))
            #z = self.model.sampling((z_mean, z_logvar)) # re-parameterize
            x_interpolate = self.model.generator(z)
            
            #plt.scatter(self.model.z, x_real_580, s=4, color='b', label='real')
            plt.scatter(self.model.z, self.model.scaler.inverse_transform(x_interpolate.numpy()), s=4, color='r', label='reconstructed')
            plt.xlim(0,1.5)
            plt.ylim(33,48)
            plt.xlabel(r'redshift $z$')
            plt.ylabel(r'distance modulus $\mu$')
            plt.legend()
            plt.show()

            pred = self.model.s_discriminator(x_interpolate)
            print(pred)
            print('Class : ',tf.argmax(pred, axis=-1))

            #x_interpolate = self.model.generator(z)

            #plt.scatter(self.model.z, self.model.scaler.inverse_transform(x_interpolate.numpy()), s=4)
            #plt.xlim(0,1.5)
            #plt.ylim(33,48)
            #plt.xlabel(r'redshift $z$')
            #plt.ylabel(r'distance modulus $\mu$')
            #plt.show()

            #pred = self.model.s_discriminator(x_interpolate)
            #print(pred)
            #print('Class : ',tf.argmax(pred, axis=-1))
