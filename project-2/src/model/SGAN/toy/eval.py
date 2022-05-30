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

from src.data.toy_models.ParabolicModel import ParabolicModel
from src.data.toy_models.SineModel import SineModel

from src.model.SGAN.Generator import Generator
from src.model.SGAN.Discriminator import Discriminator
from src.model.SGAN.UnsupervisedDiscriminator import UnsupervisedDiscriminator
from src.model.SGAN.SupervisedDiscriminator import SupervisedDiscriminator
from src.model.SGAN.SGAN import SGAN
from src.model.SGAN.toy.GANMonitor import GANMonitor

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

def generate_real_samples(train_size, z, input_dim):
    half_train_size = int(train_size/2) 
    parabolicModelObj = ParabolicModel()
    x1 = np.zeros([half_train_size, input_dim], dtype='float32')
    for i in range(half_train_size):
        x1[i] = parabolicModelObj.sample(z)
    y1 = np.zeros(half_train_size)
    sineModelObj = SineModel()
    x2 = np.zeros([half_train_size, input_dim], dtype='float32')
    for i in range(half_train_size):
        x2[i] = sineModelObj.sample(z)
    y2 = np.ones(half_train_size)
    x_real = np.concatenate([x1, x2])
    y_real = np.concatenate([y1, y2])
    return x_real, y_real


z = np.genfromtxt('../../../data/toy_models/z.csv')
x_real, y_real = generate_real_samples(train_size, z, output_dim) 
y_real = y_real[:, np.newaxis]

# split into test, validation, and training sets
x_train, x_test, y_train, y_test = train_test_split(x_real, y_real, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

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

generator = Generator(latent_dim)
discriminator = Discriminator(output_dim)
sup_discriminator = SupervisedDiscriminator(n_classes, discriminator)
unsup_discriminator = UnsupervisedDiscriminator(discriminator)

sgan = SGAN(latent_dim, generator, sup_discriminator, unsup_discriminator, z)
sgan.compile(
    tf.keras.optimizers.Adam(learning_rate=lr_gen, beta_1=beta_1_gen),
    tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=beta_1_disc), 
    tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=beta_1_disc),
    tf.keras.losses.BinaryCrossentropy(from_logits=True),
    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

sgan_checkpoint_path = os.path.join(outdir,"ckpt/sgan.ckpt")
g_checkpoint_path = os.path.join(outdir,"ckpt/generator.ckpt")
d_checkpoint_path = os.path.join(outdir,"ckpt/discriminator.ckpt")

# load the best model
sgan.load_weights(sgan_checkpoint_path)

x = sgan.generator(tf.random.normal((1,latent_dim)))

plt.figure()
plt.scatter(z, np.reshape(x, (output_dim) ), s=2)
plt.title('Generated output')
plt.xlabel('z')
plt.ylabel('x')
plt.xlim(0,1.2)
plt.ylim(0,1.2)
plt.savefig(os.path.join(outdir, 'fig/sample_out.png'))
plt.draw()
plt.pause(0.001)
plt.show()

# get the class label assigned
print(sgan.s_discriminator(x))
print('Class :', tf.argmax(sgan.s_discriminator(x), axis=-1))

input('Press enter to continue...')
