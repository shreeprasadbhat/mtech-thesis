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
z_obs = np.genfromtxt('../../../data/toy_models/z_obs.csv')
x_real, y_real = generate_real_samples(train_size, z, output_dim) 
prng = np.random.RandomState(123)
idx = prng.randint(0, output_dim, input_dim)
x_real_580 = x_real[:,idx]
y_real = y_real[:, np.newaxis]

# split into test, validation, and training sets
x_train_580, x_test_580, x_train, x_test, y_train, y_test = train_test_split(x_real_580, x_real, y_real, test_size=0.2)
x_train_580, x_val_580, x_train, x_val, y_train, y_val = train_test_split(x_train_580, x_train, y_train, test_size=0.2)

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

encoder = Encoder(input_dim, latent_dim)
generator = Generator(latent_dim)
discriminator = Discriminator(output_dim, n_classes)
sup_discriminator = SupervisedDiscriminator(discriminator)
unsup_discriminator = UnsupervisedDiscriminator(discriminator)

vaegan = VAEGAN(latent_dim, encoder, generator, discriminator, sup_discriminator, unsup_discriminator, z)
vaegan.compile(
    tf.keras.optimizers.Adam(learning_rate=lr_gen, beta_1=beta_1_gen),
    tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=beta_1_disc), 
    tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=beta_1_disc),
    tf.keras.optimizers.Adam(learning_rate=lr_gen, beta_1=beta_1_gen),
    tf.keras.losses.BinaryCrossentropy(from_logits=False),
    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
)

vaegan_checkpoint_path = os.path.join(outdir,"ckpt/vaegan.ckpt")
#g_checkpoint_path = os.path.join(outdir,"ckpt/generator.ckpt")
#d_checkpoint_path = os.path.join(outdir,"ckpt/discriminator.ckpt")

csvlogger_path = os.path.join(outdir,"log/log.txt")

callbacks = [
    #ModelCheckpoint(checkpoint_path, monitor='val_s_loss', save_best_only=True, save_weights_only=True, verbose=1),
    #EarlyStopping(monitor='val_s_loss', patience=patience, verbose=1),
    CSVLogger(csvlogger_path, separator=',', append=True),
    GANMonitor()
]

history = vaegan.fit(
    train_dataset, 
    epochs=epochs, 
    validation_data = val_dataset,
    callbacks = callbacks
)

#generator.save_weights(g_checkpoint_path)
#discriminator.save_weights(d_checkpoint_path)
vaegan.save_weights(vaegan_checkpoint_path)

# load the best model
vaegan.load_weights(vaegan_checkpoint_path)

# number of epochs, early stopped or not
epochs = len(history.history['s_loss'])

plt.figure()
plt.plot(history.history['e_loss'], '-', label='e_loss')
plt.plot(history.history['val_e_loss'], '--', label='val_e_loss')
plt.plot(history.history['g_loss'], '-', label='g_loss')
plt.plot(history.history['val_g_loss'], '--', label='val_g_loss')
plt.plot(history.history['u_loss'], '-', label='u_loss')
plt.plot(history.history['val_u_loss'], '--', label='val_u_loss')
plt.plot(history.history['s_loss'], label='s_loss')
plt.plot(history.history['val_s_loss'], label='val_s_loss')
plt.title('VAE loss curve')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig(os.path.join(outdir,'fig/loss_curve.png'))
plt.draw()
plt.pause(0.001)
#plt.show()

plt.figure()
plt.plot(history.history['s_acc'], label='s_acc')
plt.plot(history.history['val_s_acc'], label='val_s_acc')
plt.plot(history.history['u_acc'], label='u_acc')
plt.plot(history.history['val_u_acc'], label='val_u_acc')
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig(os.path.join(outdir,'fig/accuracy.png'))
plt.draw()
plt.pause(0.001)
#plt.show()


x = vaegan.mypredict(x_test_580[0].reshape(-1,580))


