import numpy as np
import matplotlib.pyplot as plt

from TrueModel import TrueModel
from SineModel import SineModel
from ParabolicModel import ParabolicModel

z = np.sort(np.random.uniform(0,1,2048))

# True Model
obj = TrueModel()

y = obj.out(z)
plt.plot(z, y, label='true model')

# Sine Model
obj = SineModel()

y = obj.out(z)
plt.plot(z, y, label='sine model')

y_max = obj.out(z, 1.1, np.pi+0.01, 0.1)
y_min = obj.out(z, 0.9, np.pi-0.01, -0.1)

plt.fill_between(z, y_max, y_min, color='b', alpha=0.1)

# Parabolic Model
obj = ParabolicModel()

y = obj.out(z)
plt.plot(z, y, label='parabolic model', color='b', alpha=0.3)

y_max = obj.out(z, -4.1, 0.01, 0.1)
y_min = obj.out(z, -3.9, -0.01, -0.1)

plt.fill_between(z, y_max, y_min)

plt.xlabel('z')
plt.ylabel('y')
plt.legend()
plt.title('distribution of output of toy models')
plt.savefig('dist_toy_models.png')

plt.show()
