import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class UnsupervisedDiscriminator(keras.Model):

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def custom_activation(self, output):
        logexpsum = tf.reduce_sum(tf.exp(output), axis=-1, keepdims=True)
        result = logexpsum / (logexpsum + 1.0)
        return result

    def call(self, inputs):
        x = self.discriminator(inputs)
        x = self.custom_activation(x)
        return x

if __name__ == '__main__':
    from Discriminator import Discriminator
    obj = UnsupervisedDiscriminator(Discriminator(2048, 3))
