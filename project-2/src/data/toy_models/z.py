import numpy as np
prng = np.random.RandomState(123)
z = prng.uniform(0, 1, 2048)
z.sort()
idx = prng.randint(0, 2048, 580)
z_obs = z[idx]

np.savetxt('z_obs.csv',z_obs)
np.savetxt('z.csv', z)
