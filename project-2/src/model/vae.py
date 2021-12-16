import tensorflow as tf

class VAE(tf.keras.Model) :
    
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.input_layer = tf.keras.InputLayer(input_shape=(4,1))
        
        # encoder
        self.conv1d1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2), activation='relu')
        self.conv1d2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2), activation='relu')
        self.conv1d3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2), activation='relu')
        self.conv1d4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2), activation='relu')
        self.dense1 = tf.keras.layers.Dense(latent_dim)

        # decoder
        self.InputLayer(input=(latent_dim))
        self.dense2 = tf.keras.layers.Dense(units = 7*7*32, activation=tf.nn.relu)
        self.reshape1 = tf.keras.layers.Reshape(target_shape=(7,7,32))
        self.conv1DTranspose1 = tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=3, strides=2, padding='same',activation='relu')
        self.conv1DTranspose2 = tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')
        self.conv1DTranspose3 = tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'(
        
 
