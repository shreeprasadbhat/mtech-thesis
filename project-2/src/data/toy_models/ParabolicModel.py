import numpy as np

class ParabolicModel:
    def __init__ (self):
        self.A_mean = -4
        self.A_err = 0.1
        self.B_mean = 0
        self.B_err = 0.01
        self.C_mean = 0
        self.C_err = 0.1
    
    def sample_A(self, size=(1,)):
        return np.random.normal(self.A_mean, self.A_err, size)

    def sample_B(self, size=(1,)):
        return np.random.normal(self.B_mean, self.B_err, size)

    def sample_C(self, size=(1,)):
        return np.random.normal(self.C_mean, self.C_err, size)
    
    def out(self, z, A=None, B=None, C=None):
        if A is None:
            A = self.A_mean
        if B is None:
            B = self.B_mean
        if C is None:
            C = self.C_mean
        return A*z*z + (-A+B)*z + C

    def err(self, z, A=None, B=None, C=None):
        if A is None:
            A = self.A_mean
        if B is None:
            B = self.B_mean
        if C is None:
            C = self.C_mean

        A_err = self.A_err
        B_err = self.B_err
        C_err = self.C_err
        
        return np.sqrt(z**4 * A_err**2 + (z**2 * (z**2 * (A_err**2 + B_err**2)) + C_err**2))
    
    def sample(self, z):
        A = self.sample_A(z.shape[0])
        B = self.sample_B(z.shape[0])
        C = self.sample_C(z.shape[0])
        return self.out(z, A, B, C)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    z = np.sort(np.random.uniform(0,1,580))

    obj = ParabolicModel()

    y = obj.out(z)
    plt.plot(z, y, color='r')

    y = obj.sample(z)
    plt.scatter(z, y, s=4)

    plt.show()
