import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# add parent directory of project to system path, to access all the packages in project, sys.path.append appends are not permanent
# add parent directory of project to system path, to access all the packages in project, sys.path.append appends are not permanent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.model.VAE.VariationalAutoEncoder import VariationalAutoEncoder

latent_dim = 1

vae = VariationalAutoEncoder(2)
vae.compile(optimizer='adam', loss='mse', metrics=['mse'])

checkpoint_path = "./saved_models/cp.ckpt"
vae.load_weights(checkpoint_path)

from matplotlib.pylab import cm
plt.figure(figsize=(10,4))
#latent_inputs = np.repeat(np.linspace(-3,3,11)[:,np.newaxis],encoding_dim,axis=1)
latent_inputs = np.random.randn(1000,1)
decoded_latent_inputs = vae.decoder(latent_inputs)
plt.scatter(decoded_latent_inputs[:,0],decoded_latent_inputs[:,1],color='b')
plt.savefig('./out/Reconstructed')
plt.show()
