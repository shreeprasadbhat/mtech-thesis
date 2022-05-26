from TrueModel import TrueModel
import numpy as np

obj = TrueModel()
z = np.random.uniform(0,1,580)
y = obj.sample(z, 580)

import matplotlib.pyplot as plt

plt.scatter(z, y, s=4, color='r')
plt.show()
