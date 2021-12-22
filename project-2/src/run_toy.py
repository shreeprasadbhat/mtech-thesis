import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split 

# add parent directory of project to system path, to access all the packages in project, sys.path.append appends are not permanent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.VariationalAutoEncoder import VariationalAutoEncoder
from data.Toy.ParabolicModel import ParabolicModel
from data.Toy.SineModel import SineModel 

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
vae.fit(x_train, x_train, batch_size=128, epochs=100, validation_data=(x_val, x_val))

encoded_test = np.array(vae.encoder(x_test))
vae_test = vae.predict(x_test)

plt.scatter(x_test[:,0],x_test[:,1],color='r',label='true signal')
plt.scatter(vae_test[:,0],vae_test[:,1],color='b',label='encoded-decoded signal')
plt.legend()
plt.show()

from matplotlib.pylab import cm
plt.figure(figsize=(10,4))
#latent_inputs = np.repeat(np.linspace(-3,3,11)[:,np.newaxis],encoding_dim,axis=1)
latent_inputs = np.random.randn(1000,1)
decoded_latent_inputs = vae.decoder(latent_inputs)
plt.scatter(decoded_latent_inputs[:,0],decoded_latent_inputs[:,1],color='b')
plt.show()
