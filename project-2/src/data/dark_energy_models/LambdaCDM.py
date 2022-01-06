import os
import numpy as np
from scipy import integrate
from numba import jit, vectorize

union = np.genfromtxt('../../../data/Union/union.txt', delimiter=' ', usecols=(1,2,3), names=True)

union.sort(order='zCMB')
z_obs = union['zCMB'].astype('float32')
mu = union['MU'].astype('float32')

prng = np.random.RandomState(123)
z = prng.uniform(0.8*np.min(z_obs), 1.2*np.max(z_obs), 1468)
z = np.concatenate((z, z_obs), axis=0)
z.sort()

omega_M_low = 0.1
omega_M_high = 0.9
# Hubble constant, hubble parameter value of present day
H0_low = 50
H0_high = 90
#speed of light in km/s 
c = 299792.458 

@jit
def hubble_parameter(z, omega_M, H0): 
    return 1 / (H0 * (omega_M * (1+z)**3 + (1-omega_M))**0.5)

@vectorize('float64(float64, float64, float64)')
def luminosity_distance(z, omega_M, H0):
    integral, integral_err = integrate.quad(hubble_parameter, 0, z, args=(omega_M, H0))
    dL =  c * (1+z) * integral
    return dL

@vectorize('float64(float64, float64, float64)')
def distance_modulus(z, omega_M, H0):
    return 5 * np.log10(luminosity_distance(z, omega_M, H0)) + 25

size = 12800 
input_dim = 2048

def lambdaCDMSample(z):
    omega_M = np.random.uniform(omega_M_low, omega_M_high, (size,1))
    H0 = np.random.uniform(H0_low, H0_high, (size,1))
    omega_M = np.tile(omega_M, (1, input_dim))
    H0 = np.tile(H0, (1, input_dim))
    return distance_modulus(z, omega_M, H0)

x_real = lambdaCDMSample(np.tile(np.reshape(z, (1, input_dim)), (size, 1)))
y_real = np.zeros(size)

idx = np.where(np.in1d(z, z_obs))[0]
x_real_580 = x_real[:, idx]

np.savetxt("x_real_580_lambdaCDM.csv", x_real_580, delimiter=",")
np.savetxt("x_real_lambdaCDM.csv", x_real, delimiter=",")
np.savetxt("y_real_lambdaCDM.csv", y_real, delimiter=",")

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    union = np.genfromtxt('../../../data/Union/union.txt', delimiter=' ', usecols=(1,2,3), names=True)
    union.sort(order='zCMB')
    z_obs = union['zCMB'].astype('float32')
    mu = union['MU'].astype('float32')

    prng = np.random.RandomState(123)
    z = prng.uniform(0.8*np.min(z_obs), 1.2*np.max(z_obs), 1468)
    z = np.concatenate((z, z_obs), axis=0)
    z.sort()
    x_real = np.genfromtxt('x_real_lambdaCDM.csv',delimiter=',')
    plt.plot(z, x_real[0])
    plt.plot(z, x_real[1])
    plt.show()
