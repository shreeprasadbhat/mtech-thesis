import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim=20):
        self.latent_dim = 20
        self.batch_size = 1 
        self.outdir = './runs/runs01/'

    def on_epoch_end(self, epoch, logs=None):
        z = tf.random.normal(shape=(1, self.latent_dim))
        x_fake = self.model.generator(z)
       
        plt.scatter(self.model.z, x_fake.numpy(), s=4)
        plt.xlim(0,1.2)
        plt.ylim(0,1.2)
        plt.show()

        print(self.model.s_discriminator(x_fake))



