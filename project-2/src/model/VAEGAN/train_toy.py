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

from VAEGAN import VAEGAN
from src.data.toy_models.ParabolicModel import ParabolicModel
from src.data.toy_models.SineModel import SineModel

readlines = ""
with open('HyperParams.json') as file:
    readlines = file.read() 

HyperParams = json.loads(readlines) 
latent_dim = HyperParams['latent_dim'] 
epochs = HyperParams['epochs']
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
idx = prng.randint(0, output_dim, input_dim)
x_real, y_real = generate_real_samples(train_size, z, output_dim) 
x_real_580 = x_real[:, idx] 
y_real = y_real[:, np.newaxis]

# split into test, validation, and training sets
x_train_580, x_test_580, x_train, x_test, y_train, y_test = train_test_split(x_real_580, x_real, y_real, test_size=0.05)
x_train_580, x_val_580, x_train, x_val, y_train, y_val = train_test_split(x_train_580, x_train, y_train, test_size=0.1)

train_dataset = ( 
    tf.data.Dataset
        .from_tensor_slices((x_train_580, x_train, y_train))
        .shuffle(buffer_size, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
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

vaegan = VAEGAN(latent_dim, input_dim, output_dim, n_classes)

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

        enc_loss, gen_loss, disc_loss, x_fake, y_fake = vaegan.train_step(x_real_580, x_real, y_real)

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
