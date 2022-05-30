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
z = np.sort(prng.uniform(0, 1, output_dim))
idx = prng.randint(0, output_dim, input_dim)
z_580 = z[idx]
x_real = generate_real_samples(train_size, z, output_dim) 
x_real_580 = x_real[:, idx] 

# split into test, validation, and training sets
x_train_580, x_test_580, x_train, x_test = train_test_split(x_real_580, x_real, test_size=0.05)
x_train_580, x_val_580, x_train, x_val = train_test_split(x_train_580, x_train, test_size=0.1)

train_dataset = ( 
    tf.data.Dataset
        .from_tensor_slices((x_train_580, x_train))
        .shuffle(buffer_size, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset
        .from_tensor_slices((x_val_580, x_val))
        .batch(batch_size)
)

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
csvlogger_path = os.path.join(outdir,"log/log.txt")

callbacks = [
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
    CSVLogger(csvlogger_path, separator=',', append=True)   
]

history = vae.fit(
    train_dataset,
    epochs=epochs, 
    validation_data=val_dataset,
    callbacks = callbacks
    )

# load the best model
vae.load_weights(checkpoint_path)

vae_test = vae.predict(x_test_580, batch_size=512)

# number of epochs, early stopped or not
epochs = len(history.history['loss'])

plt.figure()
plt.plot(np.arange(1, epochs+1, 1), history.history['loss'], label='train loss')
plt.plot(np.arange(1, epochs+1, 1), history.history['val_loss'], label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.title('Encoder loss')
plt.savefig(os.path.join(outdir,'fig/encoder_loss.png'))
plt.draw()
plt.pause(0.001)
#plt.show()

plt.figure()
plt.plot(np.arange(1, epochs+1, 1), history.history['mse'], label='train mse')
plt.plot(np.arange(1, epochs+1, 1), history.history['val_mse'], label='val mse')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.legend()
plt.title('Encoder MSE')
plt.savefig(os.path.join(outdir,'fig/encoder_mse.png'))
plt.draw()
plt.pause(0.001)
#plt.show()

#plt.figure()
#plt.scatter(z_580, x_test_580[0], s=1, color='r', label='observation')
#plt.scatter(z, vae_test[0], color='b', s=1, label='reconstruct')
#plt.xlabel('redshift z')
#plt.ylabel('Distance modulus')
#plt.legend()
#plt.title('Reconstruction of the distance modulus by the network')
#plt.savefig(os.path.join(outdir,'fig/true_vs_reconstructed'))
#plt.draw()
#plt.pause(0.001)
#plt.show()

input("Press Enter to continue...")
