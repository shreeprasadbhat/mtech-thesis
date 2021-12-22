import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split 
import tensorflow as tf
import matplotlib.pyplot as plt

# add parent directory of project to system path, to access all the packages in project, sys.path.append appends are not permanent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.model.GAN.GAN import GAN, Generator, Discriminator
from src.data.toy_models.ParabolicModel import ParabolicModel
from src.data.toy_models.SineModel import SineModel 

def generate_real_samples(size):
    parabolicModelObj = ParabolicModel()
    z1 = np.random.uniform(0,1,int(size/2))
    x1 = parabolicModelObj.sample(z1,int(size/2))
    #x1 = parabolicModelObj.out(z)
    sineModelObj = SineModel()
    z2 = np.random.uniform(0,1,int(size/2))
    x2 = sineModelObj.sample(z2, int(size/2))
    x_real = np.zeros((size,2))
    x_real[:,0] = np.concatenate([z1,z2])
    x_real[:,1] = np.concatenate([x1,x2])
    y_real = np.ones((size,1))
    return x_real, y_real

def generate_latent_samples(batch_size):
    return np.random.randn(batch_size, latent_dim)

def generate_fake_samples(n):
    latent_samples = generate_latent_samples(n)
    x_fake = generator.predict(latent_samples)
    y_fake = np.zeros(n)
    return x_fake, y_fake
   
latent_dim = 5
epochs = 100
batch_size = 128
input_size = 2

generator = Generator(input_size)
discriminator = Discriminator()
gan_model = GAN(generator, discriminator)

discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
    )

gan_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
    )

def summarize_performance(epoch, generator, discriminator, n=1000):
	# prepare real samples
	x_real, y_real = generate_real_samples(n)
	# evaluate discriminator on real examples
	_, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(n)
	# evaluate discriminator on fake examples
	_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print(epoch, acc_real, acc_fake)
	
	# scatter plot real and fake data points
	plt.scatter(x_real[:, 0], x_real[:, 1], color='red')
	plt.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
	plt.xlim([0,1])
	plt.show()

def train(generator, discriminator, gan_model, n_eval=100) :

    for i in range(epochs) :
        
        x_real, y_real = generate_real_samples(batch_size)
        x_fake, y_fake = generate_fake_samples(batch_size)

        discriminator.train_on_batch(x_real, y_real)
        discriminator.train_on_batch(x_fake, y_fake)

        x_gan = generate_latent_samples(batch_size)
        y_gan = np.ones(batch_size)

        gan_model.train_on_batch(x_gan, y_gan)

        if (i+1) % n_eval == 0:
            summarize_performance(i, generator, discriminator)


checkpoint_path = "./saved_models/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

train(generator, discriminator, gan_model)

summarize_performance(100, generator, discriminator)