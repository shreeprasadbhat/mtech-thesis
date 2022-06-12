
import numpy as np

x_real_580 = np.genfromtxt("x_real_580_CPL.csv", delimiter=",")
x_real = np.genfromtxt("x_real_CPL.csv",  delimiter=",")
y_real = np.genfromtxt("y_real_CPL.csv", delimiter=",")

cov_obs = np.genfromtxt('../../../data/Union/SCPUnion2.1_covmat_sys.txt')

err_obs = np.random.multivariate_normal(np.zeros(580), cov_obs, size=(12800,))

x_real_580_with_err = x_real_580 + err_obs

np.savetxt("x_real_580_CPL_with_err.csv", x_real_580_with_err, delimiter=",")


