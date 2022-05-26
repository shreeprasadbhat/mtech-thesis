import numpy as np

class TrueModel:
    def __init__(self):
        self.A_mean = -3.5
        self.A_err = 0.1
        self.B_mean = 3.6 
        self.B_err = 0.1
        self.C_mean = -0.1 
        self.C_err = 0.1

    def out(self, z, A=None, B=None, C=None):
        if A is None:
            A = self.A_mean
        if B is None:
            B = self.B_mean
        if C is None:
            C = self.C_mean
        return A * z**2 + B * z + C

    def sample(self, z, size=(1,), A=None, B=None, C=None):
        if A is None:
            A = np.random.normal(self.A_mean, self.A_err, size)
        if B is None:
            B = np.random.normal(self.B_mean, self.B_err, size)
        if C is None:
            C = np.random.normal(self.C_mean, self.C_err, size)
        return self.out(z, A, B, C)

if __name__ == "__main__":
    obj = TrueModel()
    z = np.random.uniform(0,1,580)
    y = obj.sample(z, 580)
    import matplotlib.pyplot as plt
    plt.scatter(z, y, s=4, color='r')
    plt.show()
