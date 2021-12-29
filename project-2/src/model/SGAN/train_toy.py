import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from numpy.random import RandomState

# add parent directory of project to system path, to access all the packages in project, sys.path.append appends are not permanent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from GAN import GAN
from src.data.toy_models.ParabolicModel import ParabolicModel
from src.data.toy_models.SineModel import SineModel

readlines = ""
with open('HyperParams.json') as file:
    readlines = file.read() 

HyperParams = json.loads(readlines) 
latent_dim = HyperParams['latent_dim'] 
epochs = HyperParams['epochs']
lr_gen = HyperParams['lr_gen']
lr_disc = HyperParams['lr_disc']
batch_size = HyperParams['batch_size']
input_dim = HyperParams['input_dim']
output_dim = HyperParams['output_dim']
n_classes = HyperParams['n_classes']
train_size = HyperParams['train_size']
buffer_size = train_size 

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

sgan = SGAN(output_dim, latent_dim, n_classes)
sgan.def_optimizer(lr_gen, lr_disc)

gen_train_loss_results = []
disc_train_loss_results = []

disc_train_acc_results = []

gen_val_loss_results = []
disc_val_loss_results = []

disc_val_acc_results = []

for epoch in range(epochs):

    gen_epoch_loss_avg = keras.metrics.Mean()
    disc_epoch_loss_avg = keras.metrics.Mean()

    disc_epoch_accuracy = keras.metrics.SparseCategoricalAccuracy()

    for x_real, y_real in train_dataset:

        gen_loss, disc_loss = sgan.train_step(x_real, y_real)

        # track progress
        gen_epoch_loss_avg.update_state(gen_loss)
        disc_epoch_loss_avg.update_state(disc_loss)
      
        disc_epoch_accuracy.update_state(y_real, sgan.discriminator.predict(x_real)[0])
            
    gen_train_loss_results.append(gen_epoch_loss_avg.result())
    disc_train_loss_results.append(disc_epoch_loss_avg.result())

    disc_train_acc_results.append(disc_epoch_accuracy.result())
    
    print('Epoch {:03d}: Gen Loss {:03g} Disc Loss {:03g} Acc {:03g}:'.format(epoch+1,
                                                                               gen_epoch_loss_avg.result(), 
                                                                               disc_epoch_loss_avg.result(),
                                                                               disc_epoch_accuracy.result()))

    gen_epoch_loss_avg = keras.metrics.Mean()
    disc_epoch_loss_avg = keras.metrics.Mean()

    disc_epoch_accuracy = keras.metrics.SparseCategoricalAccuracy()


    for x_real, y_real in val_dataset:
       
        gen_loss, disc_loss= sgan.sgan_loss(x_real, y_real, x_real.shape[0])

        # track progress
        gen_epoch_loss_avg.update_state(gen_loss)
        disc_epoch_loss_avg.update_state(disc_loss)
      
        disc_epoch_accuracy.update_state(y_real, sgan.discriminator.predict(x_real)[0])

    
    gen_val_loss_results.append(gen_epoch_loss_avg.result())
    disc_val_loss_results.append(disc_epoch_loss_avg.result())

    disc_val_acc_results.append(disc_epoch_accuracy.result())
    
    print('Epoch {:03d}: Gen Loss {:03g} Disc Loss {:03g} Acc {:03g}:'.format(epoch+1,
                                                                               gen_epoch_loss_avg.result(), 
                                                                               disc_epoch_loss_avg.result(),
                                                                               disc_epoch_accuracy.result()))

    
checkpoint_path = "./saved_models/cp.ckpt"

# Save the weights using the `checkpoint_path` format
sgan.save_weights(checkpoint_path)

plt.plot(np.arange(epochs), gen_train_loss_results, '-', label='generator train loss')
plt.plot(np.arange(epochs), gen_val_loss_results, '--', label='generator val loss')
plt.plot(np.arange(epochs), disc_train_loss_results, '-', label='discriminator train loss')
plt.plot(np.arange(epochs), disc_val_loss_results, '--', label='discriminator val loss')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.title('epoch vs loss')
plt.savefig('./out/epoch_vs_loss')
plt.show()

plt.plot(np.arange(epochs), disc_train_acc_results, '-', label='discriminator train accuracy')
plt.plot(np.arange(epochs), disc_val_acc_results, '--', label='discriminator val accuracy')

plt.xlabel('epochs')
plt.ylabel('metrics')
plt.legend()
plt.title('epoch vs metrics')
plt.savefig('./out/epoch_vs_metrics')
plt.show()
