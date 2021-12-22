import tensorflow as tf
from tensorflow.keras import layers

class Generator(tf.keras.Model):
    
    def __init__(self, n_inputs):
        super(Generator, self).__init__()
        self.dense = layers.Dense(15, activation='relu', kernel_initializer='he_uniform')
        self.out = layers.Dense(n_inputs, activation='linear')

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.out(x)
        return x

class Discriminator(tf.keras.Model):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense = layers.Dense(25, activation='relu', kernel_initializer='he_uniform')
        self.out = layers.Dense(1, activation='softmax')
    
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.out(x)
        return x

class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def call(self, inputs):
        self.discriminator.trainable = False
        x = self.generator(inputs)
        return self.discriminator(x)

if __name__ == '__main__':
    generator = Generator(2)
    discriminator = Discriminator()
    gan_model = GAN(generator, discriminator)
    
