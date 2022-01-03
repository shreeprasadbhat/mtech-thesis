import tensorflow as tf
from tensorflow import keras
from keras import layers

class Encoder(layers.Layer):
    def __init__(self, input_dim, latent_dim, filter_size=8, kernel_initializer='he_uniform', alpha=0.3):
        super(Encoder, self).__init__()
        self.reshape = layers.Reshape((input_dim, 1))
        self.conv1 = layers.Conv1D(filter_size, kernel_size=69, strides=1, kernel_initializer=kernel_initializer, use_bias=False)
        self.leakyrelu1 = layers.LeakyReLU(alpha)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filter_size*2, kernel_size=7, strides=4, kernel_initializer=kernel_initializer, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.leakyrelu2 = layers.LeakyReLU(alpha)
        self.conv3 = layers.Conv1D(filter_size*4, kernel_size=7, strides=4, kernel_initializer=kernel_initializer, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.leakyrelu3 = layers.LeakyReLU(alpha)
        self.conv4 = layers.Conv1D(filter_size*8, kernel_size=7, strides=4, kernel_initializer=kernel_initializer, use_bias=False)
        self.bn4 = layers.BatchNormalization()
        self.leakyrelu4 = layers.LeakyReLU(alpha)
        self.dense = layers.Dense(256, kernel_initializer=kernel_initializer)
        self.leakyrelu5 = layers.LeakyReLU(alpha)
        self.out = layers.Dense(latent_dim*2)        

    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyrelu4(x)
        x = layers.Flatten() (x)
        x = self.dense(x)
        x = self.leakyrelu5(x)
        z = self.out(x) 
        z_mean, z_log_var = tf.split(z, num_or_size_splits=2, axis=1)
        return z_mean, z_log_var 

if __name__ == "__main__":
    encoder = Encoder(5, 580)
