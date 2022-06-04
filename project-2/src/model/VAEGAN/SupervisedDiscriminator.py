from tensorflow import keras
from tensorflow.keras import layers

class SupervisedDiscriminator(keras.Model):

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator
        self.softmax = layers.Softmax() 

    def call(self, inputs):
        x = self.discriminator(inputs)
        x = self.softmax(x)
        return x

if __name__ == '__main__':
    from Discriminator import Discriminator
    SupervisedDiscriminator(Discriminator(2048, 3))
