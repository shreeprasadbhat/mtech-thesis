import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split 
import tensorflow as tf
import matplotlib.pyplot as plt

# add parent directory of project to system path, to access all the packages in project, sys.path.append appends are not permanent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.model.VAE.VariationalAutoEncoder import VariationalAutoEncoder
from src.data.toy_models.ParabolicModel import ParabolicModel
from src.data.toy_models.SineModel import SineModel 

# prepare training samples by sampling from prior of toymodels
parabolicModelObj = ParabolicModel()
z = np.linspace(0,1,12800)
x1 = parabolicModelObj.sample(z, 12800)
sineModelObj = SineModel()
x2 = sineModelObj.sample(z, 12800)
x = np.zeros((25600,2))
x[:,0] = np.concatenate([z,z])
x[:,1] = np.concatenate([x1,x2])


# split into test, validation, and training sets
x_temp, x_test, _, _ = train_test_split(x, x, test_size=0.05)
x_train, x_val, _, _ = train_test_split(x_temp,
                                          x_temp,
                                          test_size=0.1)
n_train = len(x_train)
n_val = len(x_val)
n_test = len(x_test)

latent_dim = 1

vae = VariationalAutoEncoder(2)
vae.compile(optimizer='adam', loss='mse', metrics=['mse'])

checkpoint_path = "./saved_models/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

vae.fit(
    x_train, x_train, 
    batch_size=128, 
    epochs=1000, 
    validation_data=(x_val, x_val),
    callbacks = [cp_callback]
    )

encoded_test = np.array(vae.encoder(x_test))
vae_test = vae.predict(x_test)

plt.scatter(x_test[:,0],x_test[:,1],color='r',label='true signal')
plt.scatter(vae_test[:,0],vae_test[:,1],color='b',label='encoded-decoded signal')
plt.legend()
plt.savefig('./out/true_vs_reconstructed')
plt.show()
