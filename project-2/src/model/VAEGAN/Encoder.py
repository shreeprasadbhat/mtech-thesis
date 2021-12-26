import tensorflow as tf
from tensorflow import keras
from keras import layers

class Encoder(keras.Model):
    def __init__(self, latent_dim, input_dim, filter_size=8):
        super(Encoder, self).__init__()
        self.reshape = layers.Reshape((input_dim, 1))
        self.conv1 = layers.Conv1D(filter_size, kernel_size=69, strides=1, kernel_initializer='he_uniform', activation=layers.LeakyReLU())
        self.conv2 = layers.Conv1D(filter_size*2, kernel_size=7, strides=4, kernel_initializer='he_uniform', activation=layers.LeakyReLU())
        self.conv3 = layers.Conv1D(filter_size*4, kernel_size=7, strides=4, kernel_initializer='he_uniform', activation=layers.LeakyReLU())
        self.conv4 = layers.Conv1D(filter_size*8, kernel_size=7, strides=4, kernel_initializer='he_uniform', activation=layers.LeakyReLU())
        self.dense = layers.Dense(256, kernel_initializer='he_uniform', activation=layers.LeakyReLU())
        self.out = layers.Dense(latent_dim*2)        

    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) 
        x = layers.Flatten() (x)
        x = self.dense(x)
        z = self.out(x) 
        z_mean, z_log_var = tf.split(z, num_or_size_splits=2, axis=1)
        return z_mean, z_log_var 

if __name__ == "__main__":
    encoder = Encoder(5, 580)
