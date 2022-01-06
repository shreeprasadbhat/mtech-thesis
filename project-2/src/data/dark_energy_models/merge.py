import numpy as np

x_real_580_lambdaCDM = np.genfromtxt('x_real_580_lambdaCDM.csv', delimiter=',')
x_real_lambdaCDM = np.genfromtxt('x_real_lambdaCDM.csv', delimiter=',')
y_real_lambdaCDM = np.genfromtxt('y_real_lambdaCDM.csv', delimiter=',')

x_real_580_omegaCDM = np.genfromtxt('x_real_580_omegaCDM.csv', delimiter=',')
x_real_omegaCDM = np.genfromtxt('x_real_omegaCDM.csv', delimiter=',')
y_real_omegaCDM = np.genfromtxt('y_real_omegaCDM.csv', delimiter=',')

x_real_580_CPL = np.genfromtxt('x_real_580_CPL.csv', delimiter=',')
x_real_CPL = np.genfromtxt('x_real_CPL.csv', delimiter=',')
y_real_CPL = np.genfromtxt('y_real_CPL.csv', delimiter=',')

x_real_580 = np.concatenate((x_real_580_lambdaCDM, x_real_580_omegaCDM, x_real_580_CPL), axis=0)
x_real = np.concatenate((x_real_lambdaCDM, x_real_omegaCDM, x_real_CPL), axis=0)
y_real = np.concatenate((y_real_lambdaCDM, y_real_omegaCDM, y_real_CPL), axis=0)

print(x_real_580.shape)
print(x_real.shape)
print(y_real.shape)

np.savetxt('x_real_580.csv', x_real_580)
np.savetxt('x_real.csv', x_real)
np.savetxt('y_real.csv', y_real)
