import os
import sys
import json
import numpy as np

from Discriminator import Discriminator
from sklearn.model_selection import train_test_split
import tensorflow as tf

# add parent directory of project to system path, to access all the packages in project, sys.path.append appends are not permanent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.data.toy_models.ParabolicModel import ParabolicModel from src.data.toy_models.SineModel import SineModel

def generate_real_samples(train_size, z, input_dim):
    half_train_size = int(train_size/2) 
    parabolicModelObj = ParabolicModel()
    x1 = np.zeros([half_train_size, input_dim])
    for i in range(half_train_size):
        x1[i] = parabolicModelObj.sample(z)
    y1 = np.zeros(half_train_size)
    sineModelObj = SineModel()
    x2 = np.zeros([half_train_size, input_dim])
    for i in range(half_train_size):
        x2[i] = sineModelObj.sample(z)
    y2 = np.ones(half_train_size)
    x_real = np.concatenate([x1, x2])
    y_real = np.concatenate([y1, y2])
    return x_real, y_real

readlines = ""
with open('HyperParams.json') as file:
    readlines = file.read() 

HyperParams = json.loads(readlines) 
latent_dim = HyperParams['latent_dim'] 
epochs = HyperParams['epochs']
lr_gen = HyperParams['lr_gen']
beta_1_gen = HyperParams['beta_1_gen']
lr_disc = HyperParams['lr_disc']
beta_1_disc= HyperParams['beta_1_disc']
batch_size = HyperParams['batch_size']
input_dim = HyperParams['input_dim']
output_dim = HyperParams['output_dim']
n_classes = HyperParams['n_classes']
train_size = HyperParams['train_size']
buffer_size = train_size 


prng = np.random.RandomState(123)
z = prng.uniform(0, 1, output_dim)
z.sort()
x_real, y_real = generate_real_samples(train_size, z, output_dim) 
y_real = y_real[:, np.newaxis]

# split into test, validation, and training sets
x_train, x_test, y_train, y_test = train_test_split(x_real, y_real, test_size=0.05)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

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

discriminator = Discriminator(output_dim, n_classes)
discriminator.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=beta_1_disc),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = 'accuracy'
    )

discriminator.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

discriminator.evaluate(test_dataset)


