import os
import sys
from shutil import copyfile
import json
import numpy as np
from sklearn.model_selection import train_test_split 
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

# add parent directory of project to system path, to access all the packages in project, sys.path.append appends are not permanent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.model.SGAN.Discriminator import Discriminator
from src.model.SGAN.SupervisedDiscriminator import SupervisedDiscriminator

readlines = ""
with open('Params.json') as file:
    readlines = file.read() 
Params = json.loads(readlines) 

latent_dim = Params['latent_dim'] 
epochs = Params['epochs']
patience = Params['patience']
lr_gen = Params['lr_gen']
beta_1_gen = Params['beta_1_gen']
lr_disc = Params['lr_disc']
beta_1_disc= Params['beta_1_disc']
batch_size = Params['batch_size']
input_dim = Params['input_dim']
output_dim = Params['output_dim']
n_classes = Params['n_classes']
train_size = Params['train_size']
buffer_size = train_size 
outdir = Params['outdir'] 

if not os.path.isdir(outdir):
    os.makedirs(outdir)
if not os.path.isdir(os.path.join(outdir, 'fig')):
    os.mkdir(os.path.join(outdir, 'fig')) 
if not os.path.isdir(os.path.join(outdir, 'ckpt')):
    os.mkdir(os.path.join(outdir, 'ckpt'))
if not os.path.isdir(os.path.join(outdir, 'log')):
    os.mkdir(os.path.join(outdir, 'log'))

copyfile('Params.json',os.path.join(outdir, 'Params.json'))

z_580 = np.genfromtxt('../../../data/dark_energy_models/z_obs.csv')
z = np.genfromtxt('../../../data/dark_energy_models/z.csv')
x_real_580 = np.genfromtxt('../../../data/dark_energy_models/x_real_580.csv')
x_real = np.genfromtxt('../../../data/dark_energy_models/x_real.csv')
y_real = np.genfromtxt('../../../data/dark_energy_models/y_real.csv')

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
        .batch(batch_size)
)

test_dataset = (
    tf.data.Dataset
        .from_tensor_slices((x_test, y_test))
        .batch(batch_size)
)

discriminator = Discriminator(output_dim)
sup_discriminator = SupervisedDiscriminator(n_classes, discriminator)

sup_discriminator.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=beta_1_disc),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = 'accuracy'
    )

checkpoint_path = os.path.join(outdir,"ckpt/cp.ckpt")
csvlogger_path = os.path.join(outdir,"log/log.txt")

callbacks = [
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
    CSVLogger(csvlogger_path, separator=',', append=True)   
]


history = sup_discriminator.fit(
        train_dataset, 
        epochs = epochs, 
        validation_data = val_dataset,
        callbacks = callbacks
        )

# load the best model
sup_discriminator.load_weights(checkpoint_path)

# number of epochs, early stopped or not
epochs = len(history.history['loss'])

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('epoch vs loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig(os.path.join(outdir,'fig/epoch_vs_loss.png'))
plt.draw()
plt.pause(0.001)
#plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Supervised accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig(os.path.join(outdir,'fig/supervised_accuracy.png'))
plt.draw()
plt.pause(0.001)
#plt.show()

sup_discriminator.evaluate(test_dataset)

input('Press enter to continue...')
