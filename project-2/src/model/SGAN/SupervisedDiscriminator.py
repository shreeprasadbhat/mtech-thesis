from tensorflow import keras
from tensorflow.keras import layers

class SupervisedDiscriminator(keras.Model):

    def __init__(self, n_classes, discriminator):
        super().__init__()
        self.discriminator = discriminator
        self.dense = layers.Dense(n_classes)

    def call(self, inputs):
        x = self.discriminator(inputs)
        x = self.dense(x)
        return x

if __name__ == '__main__':
    from Discriminator import Discriminator
    SupervisedDiscriminator(3, Discriminator(2048))
