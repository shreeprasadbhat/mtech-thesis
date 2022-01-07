import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# add parent directory of project to system path, to access all the packages in project, sys.path.append appends are not permanent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.model.VAE.VariationalAutoEncoder import VariationalAutoEncoder
from src.model.SGAN.Generator import Generator
from src.model.SGAN.Discriminator import Discriminator
from src.model.SGAN.UnsupervisedDiscriminator import UnsupervisedDiscriminator
from src.model.SGAN.SupervisedDiscriminator import SupervisedDiscriminator
from src.model.SGAN.SGAN import SGAN

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

vae = VariationalAutoEncoder(input_dim, latent_dim)
vae.compile(
        loss='mse', 
        metrics=['mse']
)

checkpoint_path = os.path.join(os.path.join('../../VAE/dark_energy', outdir),"ckpt/cp.ckpt")

# load the best model
vae.load_weights(checkpoint_path)

generator = Generator(latent_dim)
discriminator = Discriminator(output_dim)
sup_discriminator = SupervisedDiscriminator(n_classes, discriminator)
unsup_discriminator = UnsupervisedDiscriminator(discriminator)

sgan = SGAN(latent_dim, generator, sup_discriminator, unsup_discriminator)
sgan.compile(
    tf.keras.optimizers.Adam(learning_rate=lr_gen, beta_1=beta_1_gen),
    tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=beta_1_disc), 
    tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=beta_1_disc),
    tf.keras.losses.BinaryCrossentropy(from_logits=True),
    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

checkpoint_path = os.path.join(outdir,"ckpt/cp.ckpt")

# load the best model
sgan.load_weights(checkpoint_path)

out = tf.nn.softmax(sgan.predict(vae.predict(np.reshape(mu, (1, 580)))))

print(out)

input("Press Enter to continue...")
