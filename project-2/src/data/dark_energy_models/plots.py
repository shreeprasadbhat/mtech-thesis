import numpy as np
import matplotlib.pyplot as plt

x_real_lambdaCDM = np.genfromtxt('x_real_lambdaCDM.csv', delimiter=',')
x_real_omegaCDM = np.genfromtxt('x_real_omegaCDM.csv', delimiter=',')
x_real_CPL = np.genfromtxt('x_real_CPL.csv', delimiter=',')
z = np.genfromtxt('z.csv')
print(x_real_lambdaCDM.shape)
plt.plot(z, x_real_lambdaCDM[0], color='b', label=r'\LambdaCDM')
plt.plot(z, x_real_lambdaCDM[1], color='b')
plt.plot(z, x_real_lambdaCDM[2], color='b')

plt.plot(z, x_real_omegaCDM[0], color='r', label=r'\omegaCDM')
plt.plot(z, x_real_omegaCDM[1], color='r')
plt.plot(z, x_real_omegaCDM[2], color='r')

plt.plot(z, x_real_CPL[0], color='k', label=r'CPL')
plt.plot(z, x_real_CPL[1], color='k')
plt.plot(z, x_real_CPL[2], color='k')

plt.xlabel('redshift z')
plt.ylabel('distance modulus')
plt.title('Examples training samples from three dark energy models')
plt.legend()
plt.savefig('training examples from LambdaCDM, omegaCDM, CPL.png')
plt.show()


