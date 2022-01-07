import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# add parent directory of project to system path, to access all the packages in project, sys.path.append appends are not permanent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.model.VAE.VariationalAutoEncoder import VariationalAutoEncoder
from src.model.SGAN.Discriminator import Discriminator
from src.model.SGAN.SupervisedDiscriminator import SupervisedDiscriminator
from src.data.toy_models.ParabolicModel import ParabolicModel
from src.data.toy_models.SineModel import SineModel 
from src.data.toy_models.TrueModel import TrueModel


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
prng = np.random.RandomState(123)
z = prng.uniform(0, 1, output_dim)
z.sort()
idx = prng.randint(0, output_dim, input_dim)
z_580 = z[idx]


trueModelObj = TrueModel()
x_obs = trueModelObj.sample(z_580)


vae = VariationalAutoEncoder(input_dim, latent_dim)
vae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_gen, beta_1=beta_1_gen),
        loss='mse', 
        metrics=['mse']
)


checkpoint_path = os.path.join(os.path.join('../../VAE/toy', outdir),"ckpt/cp.ckpt")

# load the best model
vae.load_weights(checkpoint_path)

discriminator = Discriminator(output_dim)
sup_discriminator = SupervisedDiscriminator(n_classes, discriminator)

sup_discriminator.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=beta_1_disc),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = 'accuracy'
    )

checkpoint_path = os.path.join(outdir,"ckpt/cp.ckpt")

# load the best model
sup_discriminator.load_weights(checkpoint_path)
vae_test = np.load('vae_test.npy', allow_pickle=True)
out = tf.nn.softmax(sup_discriminator.predict(vae_test))

print(out)

input("Press Enter to continue...")
