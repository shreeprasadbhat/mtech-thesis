import os
import sys
from shutil import copyfile
import json
import numpy as np
from sklearn.model_selection import train_test_split 
import tensorflow as tf
import matplotlib.pyplot as plt

# add parent directory of project to system path, to access all the packages in project, sys.path.append appends are not permanent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.model.VAE.VariationalAutoEncoder import VariationalAutoEncoder

readlines = ""
with open('Params.json') as file:
    readlines = file.read() 
Params = json.loads(readlines) 

latent_dim = Params['latent_dim'] 
epochs = Params['epochs']
patience = Params['patience']
lr = Params['lr']
beta_1 = Params['beta_1']
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1), 
        loss='mse', 
        metrics=['mse']
)

checkpoint_path = os.path.join(outdir,"ckpt/cp.ckpt")

# load the best model
vae.load_weights(checkpoint_path)

vae_test = vae.predict(np.reshape(mu, (1,580)))

plt.scatter(z_obs, mu, s=4, color='r', label='observed')
plt.plot(z, np.reshape(vae_test, 2048), color='b',label='reconstructed')

plt.xlabel('redshift z')
plt.ylabel('Distance modulus')
plt.legend()
plt.title('Reconstruction of the distance modulus by the VAE-GAN')
plt.savefig(os.path.join(outdir,'fig/true_vs_reconstructed'))
plt.draw()
plt.pause(0.001)
#plt.show()

input("Press Enter to continue...")
