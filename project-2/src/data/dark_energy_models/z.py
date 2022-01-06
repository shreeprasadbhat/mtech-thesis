import numpy as np

union = np.genfromtxt('../../../data/Union/union.txt', delimiter=' ', usecols=(1,2,3), names=True)

union.sort(order='zCMB')
z_obs = union['zCMB'].astype('float32')
mu = union['MU'].astype('float32')

prng = np.random.RandomState(123)
z = prng.uniform(0.8*np.min(z_obs), 1.2*np.max(z_obs), 1468)
z = np.concatenate((z, z_obs), axis=0)
z.sort()

np.savetxt('z_obs.csv',z_obs)
np.savetxt('z.csv', z)
