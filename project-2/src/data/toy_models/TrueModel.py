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

    
if __name__ == "__main__":
    obj = TrueModel()
    z = np.random.uniform(0,1,580)
    y = obj.sample(z)
    import matplotlib.pyplot as plt
    plt.scatter(z, y, s=4, color='r')
    plt.show()
