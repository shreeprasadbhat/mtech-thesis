from tensorflow import keras
from keras import layers

class Discriminator(keras.Model):
    def __init__(self, n_classes, filter_size=8):
        super(Discriminator, self).__init__()
        self.reshape = layers.Reshape((2048,1))
        self.conv1 = layers.Conv1D(filter_size, kernel_size=69, strides=1, kernel_initializer='he_uniform', activation=layers.LeakyReLU())
        self.conv2 = layers.Conv1D(filter_size*2, kernel_size=7, strides=4, kernel_initializer='he_uniform', activation=layers.LeakyReLU())
        self.conv3 = layers.Conv1D(filter_size*4, kernel_size=7, strides=4, kernel_initializer='he_uniform', activation=layers.LeakyReLU())
        self.conv4 = layers.Conv1D(filter_size*8, kernel_size=7, strides=4, kernel_initializer='he_uniform', activation=layers.LeakyReLU())
        self.out = layers.Dense(n_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.reshape(inputs) 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = layers.Flatten() (x)
        return self.out(x), x

if __name__ == "__main__":
    discrminator = Discriminator(3)
