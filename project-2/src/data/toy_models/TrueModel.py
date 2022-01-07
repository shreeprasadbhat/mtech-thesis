import numpy as np

class TrueModel:
    def __init__(self):
        self.A_mean = -4
        self.A_err = 0.1
        self.B_mean = 0
        self.B_err = 0.01
        self.C_mean = 0
        self.C_err = 0.1

    def out(self, z):
        return -3.5 * z**2 + 3.6 * z - 0.1

    def err(self, z, z_err):
        return 0

    def sample(self, z):
        A = np.random.normal(self.A_mean, self.A_err, size=z.shape)
        B = np.random.normal(self.B_mean, self.B_err, size=z.shape)
        C = np.random.normal(self.C_mean, self.C_err, size=z.shape)
        return A*(z**2) + (-A+B)*z + C

if __name__ == "__main__":
    obj = TrueModel()
    z = np.random.uniform(0,1,580)
    y = obj.sample(z)
    import matplotlib.pyplot as plt
    plt.scatter(z, y, s=4, color='r')
    plt.show()
