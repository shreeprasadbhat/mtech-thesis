from TrueModel import TrueModel
from SineModel import SineModel
from ParabolicModel import ParabolicModel

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(123)
z = np.sort(rng.uniform(0,1,580))

obj = TrueModel()
y = obj.out(z)

plt.plot(z, y, color='r', label='True Model')

obj = SineModel()
y = obj.out(z)

plt.plot(z, y, color='b', label='Sine Model')

obj = ParabolicModel()
y = obj.out(z)

plt.plot(z, y, color='g', label='ParabolicModel')
plt.legend()

plt.show()
