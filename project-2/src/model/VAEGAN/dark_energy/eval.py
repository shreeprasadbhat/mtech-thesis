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

from src.model.VAEGAN.Encoder import Encoder 
from src.model.VAEGAN.Generator import Generator
from src.model.VAEGAN.Discriminator import Discriminator
from src.model.VAEGAN.UnsupervisedDiscriminator import UnsupervisedDiscriminator
from src.model.VAEGAN.SupervisedDiscriminator import SupervisedDiscriminator
from src.model.VAEGAN.VAEGAN import VAEGAN
from src.model.VAEGAN.toy.GANMonitor import GANMonitor

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

union = np.genfromtxt('../../../../data/Union/union.txt', delimiter=' ', usecols=(1,2,3), names=True)
union.sort(order='zCMB')
z_obs = union['zCMB'].astype('float32')
mu = union['MU'].astype('float32')
z = np.genfromtxt('../../../data/dark_energy_models/z.csv')

encoder = Encoder(input_dim, latent_dim)
generator = Generator(latent_dim)
discriminator = Discriminator(output_dim, n_classes)
sup_discriminator = SupervisedDiscriminator(discriminator)
unsup_discriminator = UnsupervisedDiscriminator(discriminator)

vaegan = VAEGAN(latent_dim, encoder, generator, discriminator, sup_discriminator, unsup_discriminator, z, scaler)
vaegan.compile(
    tf.keras.optimizers.Adam(learning_rate=lr_gen, beta_1=beta_1_gen),
    tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=beta_1_disc), 
    tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=beta_1_disc),
    tf.keras.optimizers.Adam(learning_rate=lr_gen, beta_1=beta_1_gen),
    tf.keras.losses.BinaryCrossentropy(from_logits=False),
    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
)

vaegan_checkpoint_path = os.path.join(outdir,"ckpt/vaegan.ckpt")

# load the best model
vaegan.load_weights(vaegan_checkpoint_path)

mu = vaegan.scaler.transform(mu.reshape(-1, 580))

x = vaegan.mypredict(mu)

input('Press enter to continue...')
