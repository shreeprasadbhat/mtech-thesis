import os
import sys
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Generator import Generator
from Discriminator import Discriminator
from UnsupervisedDiscriminator import UnsupervisedDiscriminator
from SupervisedDiscriminator import SupervisedDiscriminator
from SGAN import SGAN

# add parent directory of project to system path, to access all the packages in project, sys.path.append appends are not permanent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.data.toy_models.ParabolicModel import ParabolicModel
from src.data.toy_models.SineModel import SineModel

def generate_real_samples(train_size, z, input_dim):
    half_train_size = int(train_size/2) 
    parabolicModelObj = ParabolicModel()
    x1 = np.zeros([half_train_size, input_dim])
    for i in range(half_train_size):
        x1[i] = parabolicModelObj.sample(z)
    y1 = np.zeros(half_train_size)
    sineModelObj = SineModel()
    x2 = np.zeros([half_train_size, input_dim])
    for i in range(half_train_size):
        x2[i] = sineModelObj.sample(z)
    y2 = np.ones(half_train_size)
    x_real = np.concatenate([x1, x2])
    y_real = np.concatenate([y1, y2])
    return x_real, y_real

readlines = ""
with open('HyperParams.json') as file:
    readlines = file.read() 

HyperParams = json.loads(readlines) 
latent_dim = HyperParams['latent_dim'] 
epochs = HyperParams['epochs']
lr_gen = HyperParams['lr_gen']
beta_1_gen = HyperParams['beta_1_gen']
lr_disc = HyperParams['lr_disc']
beta_1_disc= HyperParams['beta_1_disc']
batch_size = HyperParams['batch_size']
input_dim = HyperParams['input_dim']
output_dim = HyperParams['output_dim']
n_classes = HyperParams['n_classes']
train_size = HyperParams['train_size']
buffer_size = train_size 


prng = np.random.RandomState(123)
z = prng.uniform(0, 1, output_dim)
z.sort()
x_real, y_real = generate_real_samples(train_size, z, output_dim) 
y_real = y_real[:, np.newaxis]

# split into test, validation, and training sets
x_train, x_test, y_train, y_test = train_test_split(x_real, y_real, test_size=0.05)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

train_dataset = ( 
    tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset
        .from_tensor_slices((x_val, y_val))
        .shuffle(buffer_size)
        .batch(batch_size)
)

test_dataset = (
    tf.data.Dataset
        .from_tensor_slices((x_test, y_test))
        .shuffle(buffer_size)
        .batch(batch_size)
)

# generate points in latent space as input for the generator
def generate_latent_points(lac_disc_optimizertent_dim, n_samples):
	return tf.random.normal((n_samples, latent_dim))
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	x = generator.predict(z_input)
	# create class labels
	y = tf.zeros((n_samples, 1))
	return x, y
 
def summarize_performance(step, generator, sup_discriminator, latent_dim, dataset, n_samples=100):
	# evaluate the classifier model
	_, acc = sup_discriminator.evaluate(dataset, verbose=0)
	print('Classifier Accuracy: %.3f%%' % (acc * 100))

generator = Generator(latent_dim)
discriminator = Discriminator(output_dim, n_classes)
sup_discriminator = SupervisedDiscriminator(discriminator)
unsup_discriminator = UnsupervisedDiscriminator(discriminator)

sup_discriminator.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=beta_1_disc), 
    metrics=['accuracy']
)
unsup_discriminator.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), 
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=beta_1_disc), 
)

sgan = SGAN(generator, unsup_discriminator)
sgan.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), 
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_gen, beta_1=beta_1_gen)
)

d_loss_avg = tf.keras.metrics.Mean()
real_loss_avg = tf.keras.metrics.Mean()
fake_loss_avg = tf.keras.metrics.Mean()
g_loss_avg = tf.keras.metrics.Mean() 

d_acc_avg = tf.keras.metrics.SparseCategoricalAccuracy()

d_train_loss_results = []
d_train_acc_results = []
g_train_loss_results = []

d_val_loss_results = []
d_val_acc_results = []
g_val_loss_results = []

for epoch in range(epochs):

    d_loss_avg.reset_states()
    real_loss_avg.reset_states()
    fake_loss_avg.reset_states()
    g_loss_avg.reset_states()

    d_acc_avg.reset_states()

    for x_real, y_real in train_dataset:
        batch_size = x_real.shape[0]
        
        d_loss, d_acc = sup_discriminator.train_on_batch(x_real, y_real)
        real_loss = unsup_discriminator.train_on_batch(x_real, tf.ones((batch_size,1)))
        x_fake, y_fake = generate_fake_samples(generator, latent_dim, batch_size)
        fake_loss = unsup_discriminator.train_on_batch(x_fake, y_fake)
        x_gan, y_gan = generate_latent_points(latent_dim, batch_size), tf.ones((batch_size, 1))
        g_loss = sgan.train_on_batch(x_gan, y_gan)

        d_loss_avg.update_state(d_loss) 
        real_loss_avg.update_state(real_loss)
        fake_loss_avg.update_state(fake_loss)
        g_loss_avg.update_state(g_loss)

        d_acc_avg.update_state(y_real, sup_discriminator.predict(x_real))

    d_train_loss_results.append(d_loss_avg.result())
    d_train_acc_results.append(d_acc_avg.result()*100)
    g_train_loss_results.append(g_loss_avg.result())

    print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (
                                                        epoch+1, 
                                                        d_loss_avg.result(), 
                                                        d_acc_avg.result()*100, 
                                                        real_loss_avg.result(), 
                                                        fake_loss_avg.result(), 
                                                        g_loss_avg.result()))
    
    d_loss_avg.reset_states()
    real_loss_avg.reset_states()
    fake_loss_avg.reset_states()
    g_loss_avg.reset_states()
    
    d_acc_avg.reset_states()
    
    for x_real, y_real in val_dataset:
        batch_size = x_real.shape[0]
        d_loss, d_acc = sup_discriminator.test_on_batch(x_real, y_real)
        real_loss = unsup_discriminator.test_on_batch(x_real, tf.ones((batch_size,1)))
        x_fake, y_fake = generate_fake_samples(generator, latent_dim, batch_size)
        fake_loss = unsup_discriminator.test_on_batch(x_fake, y_fake)
        x_gan, y_gan = generate_latent_points(latent_dim, batch_size), tf.ones((batch_size, 1))
        g_loss = sgan.test_on_batch(x_gan, y_gan)

        # evaluate the model performance every so often
        d_loss_avg.update_state(d_loss) 
        real_loss_avg.update_state(real_loss)
        fake_loss_avg.update_state(fake_loss)
        g_loss_avg.update_state(g_loss)

        d_acc_avg.update_state(y_real, sup_discriminator.predict(x_real))

    d_val_loss_results.append(d_loss_avg.result())
    d_val_acc_results.append(d_acc_avg.result()*100)
    g_val_loss_results.append(g_loss_avg.result())

    print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (epoch+1, 
                                                        d_loss_avg.result(), 
                                                        d_acc_avg.result()*100, 
                                                        real_loss_avg.result(), 
                                                        fake_loss_avg.result(), 
                                                        g_loss_avg.result()))


checkpoint_path = "./saveunsup_discriminators/cp.ckpt"

# Save the weights using the `checkpoint_path` format
sgan.save_weights(checkpoint_path)

plt.plot(np.arange(epochs), g_train_loss_results, '-', label='generator train loss')
plt.plot(np.arange(epochs), g_val_loss_results, '--', label='generator val loss')
plt.plot(np.arange(epochs), d_train_loss_results, '-', label='discriminator train loss')
plt.plot(np.arange(epochs), d_val_loss_results, '--', label='discriminator val loss')


plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.title('epoch vs loss')
plt.savefig('./out/epoch_vs_loss')
plt.show()

plt.plot(np.arange(epochs), d_train_acc_results, '-', label='discriminator train accuracy')
plt.plot(np.arange(epochs), d_val_acc_results, '--', label='discriminator val accuracy')

plt.xlabel('epochs')
plt.ylabel('metrics')
plt.legend()
plt.title('epoch vs metrics')
plt.savefig('./out/epoch_vs_metrics')
plt.show()
