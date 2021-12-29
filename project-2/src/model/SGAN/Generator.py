from tensorflow import keras
from keras import layers

class Generator(keras.Model):
    def __init__(self, latent_dim, kernel_initializer='he_uniform', alpha=0.): 
        super(Generator, self).__init__()
        self.reshape = layers.Reshape((latent_dim,1))
        self.dense = layers.Dense(128, kernel_initializer=kernel_initializer, activation=layers.LeakyReLU(alpha))
        self.conv1 = layers.Conv1DTranspose(64, kernel_size=7, strides=4, kernel_initializer=kernel_initializer, activation=layers.LeakyReLU(alpha), padding='same')
        self.conv2 = layers.Conv1DTranspose(32, kernel_size=7, strides=4, kernel_initializer=kernel_initializer, activation=layers.LeakyReLU(alpha), padding='same')
        self.conv3 = layers.Conv1DTranspose(16, kernel_size=7, strides=4, kernel_initializer=kernel_initializer, activation=layers.LeakyReLU(alpha), padding='same')
        self.conv4 = layers.Conv1DTranspose(1, kernel_size=69, strides=2, kernel_initializer=kernel_initializer, activation='tanh', padding='same')
         
    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.dense(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x 

if __name__ == "__main__":
    generator = Generator(2)
