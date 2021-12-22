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

from src.model.VAEGAN.VAEGAN import VAEGAN
from src.data.toy_models.ParabolicModel import ParabolicModel
from src.data.toy_models.SineModel import SineModel 
from src.data.toy_models.TrueModel import TrueModel

def generate_real_samples(size, z):
    parabolicModelObj = ParabolicModel() 
    x1 = np.zeros([int(size/2), 2048]) 
    for i in range(int(size/2)):
        x1[i] = parabolicModelObj.sample(z)     
    y1 = tf.zeros(int(size/2))
    sineModelObj = SineModel()
    x2 = np.zeros([int(size/2), 2048])
    for i in range(int(size/2)):
        x2[i] = sineModelObj.sample(z)
    y2 = tf.ones(int(size/2))
    x_real = np.concatenate([x1, x2])
    y_real = np.concatenate([y1, y2])
    return x_real, y_real 

def generate_latent_samples(batch_size):
    return np.random.randn(batch_size, latent_dim)

def generate_fake_samples(n):
    latent_samples = generate_latent_samples(n)
    x_fake = generator.predict(latent_samples)
    y_fake = np.zeros(n)
    return x_fake, y_fake

#HyperParams = json.load('HyperParams.json')
latent_dim = 16 
epochs = 1000
batch_size = 512 
input_size = 2048
n_classes = 3
train_size = 26000 
buffer_size = train_size 

prng = RandomState(123)
# generate z
idx = prng.randint(0,2048,580)
z = prng.uniform(0,1,2048)
z.sort()

x_real, y_real = generate_real_samples(train_size, z)
#x_real_580 = x_real[:,:580]
x_real_580 = x_real[:,idx]
print(x_real_580.shape)

x_real_580 = np.reshape(x_real_580, (-1,580,1))
# split into test, validation, and training sets

x_train = x_real[:22230]
x_train_580 = x_real_580[:22230]
y_train = y_real[:22230]
x_val = x_real[22230:24700]
x_val_580 = x_real_580[22230:24700]
y_val = y_real[22230:24700]
x_test = x_real[24700:]
x_test_580 = x_real[24700:]
y_test = y_real[24700:]

#x_train, x_test, y_train, y_test = train_test_split(x_real, y_real, test_size=0.05)
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

train_dataset = ( 
    tf.data.Dataset
        .from_tensor_slices((x_train_580, x_train, y_train))
        .shuffle(buffer_size, reshuffle_each_iteration=True)
        .batch(batch_size)
)
val_dataset = (
    tf.data.Dataset
        .from_tensor_slices((x_val_580, x_val, y_val))
        .shuffle(buffer_size)
        .batch(batch_size)
)
test_dataset = (
    tf.data.Dataset
        .from_tensor_slices((x_test_580, x_test, y_test))
        .shuffle(buffer_size)
        .batch(batch_size)
)

vaegan = VAEGAN(latent_dim, input_size, n_classes)

enc_optimizer = keras.optimizers.Adam()
gen_optimizer = keras.optimizers.Adam()
disc_optimizer = keras.optimizers.Adam()

enc_train_loss_results = []
gen_train_loss_results = []
disc_train_loss_results = []

enc_train_mse_results = []
gen_train_acc_results = []
disc_train_acc_results = []

enc_val_loss_results = []
gen_val_loss_results = []
disc_val_loss_results = []

enc_val_mse_results = []
gen_val_acc_results = []
disc_val_acc_results = []

for epoch in range(epochs):

    enc_epoch_loss_avg = keras.metrics.Mean()
    gen_epoch_loss_avg = keras.metrics.Mean()
    disc_epoch_loss_avg = keras.metrics.Mean()

    enc_epoch_mse = keras.metrics.MeanSquaredError()
    gen_epoch_accuracy = keras.metrics.SparseCategoricalAccuracy()
    disc_epoch_accuracy = keras.metrics.SparseCategoricalAccuracy()

    for x_real_580, x_real, y_real in train_dataset:
        with tf.GradientTape() as enc_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            enc_loss, gen_loss, disc_loss, x_fake, y_fake = vaegan.vaegan_loss(x_real_580, x_real, y_real)      
            
        # calculate gradients
        enc_grads = enc_tape.gradient(enc_loss, vaegan.encoder.trainable_variables)
        gen_grads = gen_tape.gradient(gen_loss, vaegan.generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, vaegan.discriminator.trainable_variables)
 
        # apply gradients
        enc_optimizer.apply_gradients(zip(enc_grads, vaegan.encoder.trainable_weights))
        gen_optimizer.apply_gradients(zip(gen_grads, vaegan.generator.trainable_weights))
        disc_optimizer.apply_gradients(zip(disc_grads, vaegan.discriminator.trainable_weights)) 

        # track progress
        enc_epoch_loss_avg.update_state(enc_loss)
        gen_epoch_loss_avg.update_state(gen_loss)
        disc_epoch_loss_avg.update_state(disc_loss)
      
        enc_epoch_mse.update_state(x_real, x_fake) 
        gen_epoch_accuracy.update_state(y_real, vaegan.discriminator.predict(x_real)[0])
        disc_epoch_accuracy.update_state(y_fake, vaegan.discriminator.predict(x_fake)[0])
    
    enc_train_loss_results.append(enc_epoch_loss_avg.result())
    gen_train_loss_results.append(gen_epoch_loss_avg.result())
    disc_train_loss_results.append(disc_epoch_loss_avg.result())

    enc_train_mse_results.append(enc_epoch_mse.result()) 
    gen_train_acc_results.append(gen_epoch_accuracy.result())
    disc_train_acc_results.append(disc_epoch_accuracy.result())
    
    print('Epoch {:03d}: Enc Loss {:03g} Mse {:03g}: Gen Loss {:03g} Acc {:03g}: Disc Loss {:03g} Acc {:03g}:'.format(epoch+1,
                                                                               enc_epoch_loss_avg.result(), 
                                                                               enc_epoch_mse.result(), 
                                                                               gen_epoch_loss_avg.result(), 
                                                                               gen_epoch_accuracy.result(),
                                                                               disc_epoch_loss_avg.result(),
                                                                               gen_epoch_accuracy.result()))

    enc_epoch_loss_avg = keras.metrics.Mean()
    gen_epoch_loss_avg = keras.metrics.Mean()
    disc_epoch_loss_avg = keras.metrics.Mean()

    enc_epoch_mse = keras.metrics.MeanSquaredError()
    gen_epoch_accuracy = keras.metrics.SparseCategoricalAccuracy()
    disc_epoch_accuracy = keras.metrics.SparseCategoricalAccuracy()


    for x_real_580, x_real, y_real in val_dataset:
       
        enc_loss, gen_loss, disc_loss, x_fake, y_fake = vaegan.vaegan_loss(x_real_580, x_real, y_real)

        # track progress
        enc_epoch_loss_avg.update_state(enc_loss)
        gen_epoch_loss_avg.update_state(gen_loss)
        disc_epoch_loss_avg.update_state(disc_loss)
      
        enc_epoch_mse.update_state(x_real, x_fake) 
        gen_epoch_accuracy.update_state(y_real, vaegan.discriminator.predict(x_real)[0])
        disc_epoch_accuracy.update_state(y_fake, vaegan.discriminator.predict(x_fake)[0])
    
    enc_val_loss_results.append(enc_epoch_loss_avg.result())
    gen_val_loss_results.append(gen_epoch_loss_avg.result())
    disc_val_loss_results.append(disc_epoch_loss_avg.result())

    enc_val_mse_results.append(enc_epoch_mse.result()) 
    gen_val_acc_results.append(gen_epoch_accuracy.result())
    disc_val_acc_results.append(disc_epoch_accuracy.result())
    
    print('Epoch {:03d}: Enc Loss {:03g} Mse {:03g}: Gen Loss {:03g} Acc {:03g}: Disc Loss {:03g} Acc {:03g}:'.format(epoch+1,
                                                                               enc_epoch_loss_avg.result(), 
                                                                               enc_epoch_mse.result(), 
                                                                               gen_epoch_loss_avg.result(), 
                                                                               gen_epoch_accuracy.result(),
                                                                               disc_epoch_loss_avg.result(),
                                                                               gen_epoch_accuracy.result()))

    
checkpoint_path = "./saved_models/cp.ckpt"

# Save the weights using the `checkpoint_path` format
vaegan.save_weights(checkpoint_path)

plt.plot(np.arange(epochs), enc_train_loss_results, '-', label='encoder train loss')
plt.plot(np.arange(epochs), enc_val_loss_results, '-', label='encoder val loss')
plt.plot(np.arange(epochs), gen_train_loss_results, '-', label='generator train loss')
plt.plot(np.arange(epochs), gen_val_loss_results, '--', label='generator val loss')
plt.plot(np.arange(epochs), disc_train_loss_results, '--', label='discriminator train loss')
plt.plot(np.arange(epochs), disc_val_loss_results, '--', label='discriminator val loss')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.title('epoch vs loss')
plt.savefig('./out/epoch_vs_loss')
plt.show()

plt.plot(np.arange(epochs), enc_train_mse_results, '-', label='encoder train mse')
plt.plot(np.arange(epochs), enc_val_mse_results, '-', label='encoder val mse')
plt.plot(np.arange(epochs), gen_train_acc_results, '-', label='generator train accuracy')
plt.plot(np.arange(epochs), gen_val_acc_results, '--', label='generator val accuracy')
plt.plot(np.arange(epochs), disc_train_acc_results, '--', label='discriminator train accuracy')
plt.plot(np.arange(epochs), disc_val_acc_results, '--', label='discriminator val accuracy')

plt.xlabel('epochs')
plt.ylabel('metrics')
plt.legend()
plt.title('epoch vs metrics')
plt.savefig('./out/epoch_vs_metrics')
plt.show()
