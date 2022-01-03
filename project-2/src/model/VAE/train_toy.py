import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split 
import tensorflow as tf
import matplotlib.pyplot as plt

# add parent directory of project to system path, to access all the packages in project, sys.path.append appends are not permanent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.model.VAE.VariationalAutoEncoder import VariationalAutoEncoder
from src.data.toy_models.ParabolicModel import ParabolicModel
from src.data.toy_models.SineModel import SineModel 

readlines = ""
with open('HyperParams.json') as file:
    readlines = file.read() 

HyperParams = json.loads(readlines) 
latent_dim = HyperParams['latent_dim'] 
epochs = HyperParams['epochs']
lr = HyperParams['lr']
beta_1 = HyperParams['beta_1']
batch_size = HyperParams['batch_size']
input_dim = HyperParams['input_dim']
output_dim = HyperParams['output_dim']
n_classes = HyperParams['n_classes']
train_size = HyperParams['train_size']
buffer_size = train_size 

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

checkpoint_path = "./saved_models/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
vae.fit(
    train_dataset,
    epochs=epochs, 
    validation_data=val_dataset,
    callbacks = [cp_callback]
    )

vae_test = vae.predict(x_test_580, batch_size=512)

plt.scatter(z_580, x_test_580[0], color='r', label='true signal')
plt.scatter(z, vae_test[0], color='b', label='interpolated signal')
plt.legend()
plt.savefig('./out/true_vs_reconstructed')
plt.show()
