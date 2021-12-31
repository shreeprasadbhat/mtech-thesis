from tensorflow import keras
from tensorflow.keras import layers

class Discriminator(layers.Layer):
    def __init__(self, input_dim, n_classes, filter_size=8, kernel_initializer='he_uniform', alpha=0.):
        super(Discriminator, self).__init__()
        self.reshape = layers.Reshape((input_dim,1))
        self.conv1 = layers.Conv1D(filter_size, kernel_size=69, strides=1, kernel_initializer=kernel_initializer, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.leakyrelu1 = layers.LeakyReLU(alpha)
        self.conv2 = layers.Conv1D(filter_size*2, kernel_size=7, strides=4, kernel_initializer=kernel_initializer, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.leakyrelu2 = layers.LeakyReLU(alpha)
        self.conv3 = layers.Conv1D(filter_size*4, kernel_size=7, strides=4, kernel_initializer=kernel_initializer, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.leakyrelu3 = layers.LeakyReLU(alpha)
        self.conv4 = layers.Conv1D(filter_size*8, kernel_size=7, strides=4, kernel_initializer=kernel_initializer, use_bias=False)
        self.bn4 = layers.BatchNormalization()
        self.leakyrelu4 = layers.LeakyReLU(alpha)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(n_classes)
    
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
        x = self.flatten(x)
        x = self.dense(x)
        return x 

if __name__ == "__main__":
    discrminator = Discriminator(2048, 2)
