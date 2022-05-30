import os
import sys
from shutil import copyfile
import json
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

# add parent directory of project to system path, to access all the packages in project, sys.path.append appends are not permanent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.model.GAN.Generator import Generator
from src.model.GAN.Discriminator import Discriminator
from src.model.GAN.GAN import GAN
from src.model.GAN.dark_energy.GANMonitor import GANMonitor

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

z = np.genfromtxt('../../../data/dark_energy_models/z.csv', dtype=np.float32)
x_real = np.genfromtxt('../../../data/dark_energy_models/x_real.csv', dtype=np.float32)

# normalize the y data, very important for GAN training
scaler = StandardScaler()
scaler.fit(x_real[0].reshape(1,-1))
x_real = scaler.transform(x_real)

generator = Generator(latent_dim)
discriminator = Discriminator(output_dim)

gan = GAN(latent_dim, generator, discriminator, z, scaler)
gan.compile(
    tf.keras.optimizers.Adam(learning_rate=lr_gen, beta_1=beta_1_gen),
    tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=beta_1_disc), 
    tf.keras.losses.BinaryCrossentropy(from_logits=True),
)

gan_checkpoint_path = os.path.join(outdir,"ckpt/gan.ckpt")
g_checkpoint_path = os.path.join(outdir,"ckpt/generator.ckpt")
d_checkpoint_path = os.path.join(outdir,"ckpt/discriminator.ckpt")

# load the best model
gan.load_weights(gan_checkpoint_path)

x = gan.generator(tf.random.normal((1,latent_dim)))
x = scaler.inverse_transform(x)
plt.figure()
plt.scatter(z, np.reshape(x, (output_dim) ), s=2)
plt.title('Generated output')
plt.xlabel('z')
plt.ylabel('x')
plt.xlim(0,1)
plt.ylim(33,50.)
plt.savefig(os.path.join(outdir, 'fig/sample_out.png'))
plt.draw()
plt.pause(0.001)
#plt.show()

input('Press enter to continue...')
