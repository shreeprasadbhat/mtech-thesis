from tensorflow import keras
from keras import layers

class Generator(keras.Model):
    def __init__(self, latent_dim, kernel_initializer='he_uniform', alpha=0.3): 
        super(Generator, self).__init__()
        self.reshape = layers.Reshape((latent_dim,1))
        self.dense = layers.Dense(128, kernel_initializer=kernel_initializer, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.leakyrelu1 = layers.LeakyReLU(alpha) 
        self.conv1 = layers.Conv1DTranspose(64, kernel_size=7, strides=4, padding='same', kernel_initializer=kernel_initializer, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.leakyrelu2 = layers.LeakyReLU(alpha) 
        self.conv2 = layers.Conv1DTranspose(32, kernel_size=7, strides=4, padding='same', kernel_initializer=kernel_initializer, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.leakyrelu3 = layers.LeakyReLU(alpha) 
        self.conv3 = layers.Conv1DTranspose(16, kernel_size=7, strides=4, padding='same', kernel_initializer=kernel_initializer, use_bias=False)
        self.bn4 = layers.BatchNormalization()
        self.leakyrelu4 = layers.LeakyReLU(alpha) 
        self.conv4 = layers.Conv1DTranspose(1, kernel_size=69, strides=2, padding='same', kernel_initializer=kernel_initializer, use_bias=False)
        self.tanh = layers.Activation('tanh')
         
    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.dense(x)
        x = self.bn1(x)
        x = self.leakyrelu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.leakyrelu2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.leakyrelu3(x)
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.leakyrelu4(x)
        x = self.conv4(x)
        x = self.tanh(x)
        return x 

if __name__ == "__main__":
    generator = Generator(2)
