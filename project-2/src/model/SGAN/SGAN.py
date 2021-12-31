import tensorflow as tf
from tensorflow import keras
from keras import layers 

from Generator import Generator
from Discriminator import Discriminator

class SGAN(keras.Model):
    def __init__(self, generator, discriminator ):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
                      
    def call(self, inputs):
        self.discriminator.trainable = False
        x = self.generator(inputs)
        x = self.discriminator(x)
        return x

if __name__ == "__main__":
    from Generator import Generator
    from Discriminator import Discriminator
    from UnsupervisedDiscriminator import UnsupervisedDiscriminator
    SGAN(Generator(2), UnsupervisedDiscriminator(Discriminator(2048,2)))
