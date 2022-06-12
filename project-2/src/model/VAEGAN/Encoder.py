import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

class Encoder(keras.Model):
    def __init__(self, 
            input_dim, 
            latent_dim,
            filter_size=32, 
            kernel_size=7, 
            strides=4, 
            kernel_initializer='he_uniform', 
            alpha=0.3, 
            kernel_regularizer=regularizers.L2(0.), 
            dropout=0.):
        super(Encoder, self).__init__()

        self.reshape = layers.Reshape((input_dim,1))

        self.conv1 = layers.Conv1D(
                                filter_size, 
                                kernel_size=69, 
                                strides=1, 
                                kernel_initializer=kernel_initializer, 
                                use_bias=False, 
                                kernel_regularizer=kernel_regularizer)
        self.bn1 = layers.BatchNormalization()
        self.leakyrelu1 = layers.LeakyReLU(alpha)
        self.dropout1 = layers.Dropout(dropout)

        self.conv2 = layers.Conv1D(
                                filter_size, 
                                kernel_size=kernel_size, 
                                strides=strides, 
                                kernel_initializer=kernel_initializer, 
                                use_bias=False, 
                                kernel_regularizer=kernel_regularizer)
        self.bn2 = layers.BatchNormalization()
        self.leakyrelu2 = layers.LeakyReLU(alpha)
        self.dropout2 = layers.Dropout(dropout)

        self.conv3 = layers.Conv1D(
                                filter_size, 
                                kernel_size=kernel_size, 
                                strides=strides, 
                                kernel_initializer=kernel_initializer, 
                                use_bias=False, 
                                kernel_regularizer=kernel_regularizer)
        self.bn3 = layers.BatchNormalization()
        self.leakyrelu3 = layers.LeakyReLU(alpha)
        self.dropout3 = layers.Dropout(dropout)

        self.conv4 = layers.Conv1D(
                                filter_size, 
                                kernel_size=kernel_size, 
                                strides=strides, 
                                kernel_initializer=kernel_initializer, 
                                use_bias=False, 
                                kernel_regularizer=kernel_regularizer)
        self.bn4 = layers.BatchNormalization()
        self.leakyrelu4 = layers.LeakyReLU(alpha)
        self.dropout4 = layers.Dropout(dropout)

        self.flatten = layers.Flatten()

        self.dense1 = layers.Dense(
                                1024, 
                                use_bias=False, 
                                kernel_regularizer=kernel_regularizer)
        self.bn5 = layers.BatchNormalization()
        self.leakyrelu5 = layers.LeakyReLU(alpha)
        self.dropout5 = layers.Dropout(dropout)

        self.dense2 = layers.Dense(512, kernel_regularizer=kernel_regularizer)
        self.leakyrelu6 = layers.LeakyReLU(alpha)

        self.dense3 = layers.Dense(latent_dim*2)

    def call(self, inputs):

        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu3(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyrelu4(x)
        x = self.dropout4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn5(x)
        x = self.leakyrelu5(x)
        x = self.dropout5(x)
        x = self.dense2(x)
        x = self.leakyrelu6(x)
        z = self.dense3(x)

        z_mean, z_log_var = tf.split(z, num_or_size_splits=2, axis=1)
        return z_mean, z_log_var 

if __name__ == "__main__":
    encoder = Encoder(580, 20) 
