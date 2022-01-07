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

from src.model.VAE.VariationalAutoEncoder import VariationalAutoEncoder
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
lr = Params['lr']
beta_1 = Params['beta_1']
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
    x1 = np.zeros([half_train_size, input_dim])
    for i in range(half_train_size):
        x1[i] = parabolicModelObj.sample(z)
    sineModelObj = SineModel()
    x2 = np.zeros([half_train_size, input_dim])
    for i in range(half_train_size):
        x2[i] = sineModelObj.sample(z)
    x_real = np.concatenate([x1, x2])
    return x_real

prng = np.random.RandomState(123)
z = prng.uniform(0, 1, output_dim)
z.sort()
idx = prng.randint(0, output_dim, input_dim)
z_580 = z[idx]
x_real = generate_real_samples(train_size, z, output_dim) 
x_real_580 = x_real[:, idx] 

# split into test, validation, and training sets
x_train_580, x_test_580, x_train, x_test = train_test_split(x_real_580, x_real, test_size=0.05)
x_train_580, x_val_580, x_train, x_val = train_test_split(x_train_580, x_train, test_size=0.1)

test_dataset = (
    tf.data.Dataset
        .from_tensor_slices((x_test_580, x_test))
        .batch(batch_size)
)

vae = VariationalAutoEncoder(input_dim, latent_dim)
vae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1), 
        loss='mse', 
        metrics=['mse']
)

checkpoint_path = os.path.join(outdir,"ckpt/cp.ckpt")

# load the best model
vae.load_weights(checkpoint_path)

trueModelObj = TrueModel()
x_obs = trueModelObj.sample(z_580)
x_obs = np.reshape(x_obs, (1, 580))
vae_test = vae.predict(x_obs)
np.save('vae_test.npy',vae_test)
plt.figure()
plt.scatter(z_580, x_obs, s=4, color='r', label='observed')
plt.plot(z, np.reshape(vae_test, 2048), color='b',label='reconstructed')
plt.plot(z, trueModelObj.out(z), color='k', label='real')

plt.xlabel('redshift z')
plt.ylabel('Distance modulus')
plt.legend()
plt.title('Reconstruction of the distance modulus by the network')
plt.savefig(os.path.join(outdir,'fig/true_vs_reconstructed'))
plt.draw()
plt.pause(0.001)
#plt.show()

input("Press Enter to continue...")
