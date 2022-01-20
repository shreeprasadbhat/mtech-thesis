import numpy as np

class SineModel:
    def __init__(self):
        self.A_mean = 1
        self.A_err = 0.1
        self.w_mean = np.pi
        self.w_err = 0.1
        self.C_mean = 0
        self.C_err = 0.1

    def sample_A(self, size=(1,)):
        return np.random.normal(self.A_mean, self.A_err, size)

    def sample_w(self, size=(1,)):
        return np.random.normal(self.w_mean, self.w_err, size)

    def sample_C(self, size=(1,)):
        return np.random.normal(self.C_mean, self.C_err, size)

    def out(self, z, A=None, w=None, C=None):
        if A is None:
            A = self.A_mean
        if w is None:
            w = self.w_mean
        if C is None:
            C = self.C_mean
        return A*np.sin(w*z) + C
    
    def err(self, z, A=None, w=None, C=None):
        if A is None:
            A = self.A_mean
        if w is None:
            w = self.w_mean
        if C is None:
            C = self.C_mean

        w_err = self.w_err
        A_err = self.A_err
        C_err = self.C_err
        
        return np.sqrt(
            (A*np.sin(w*z))**2 * 
            ((A_err / A)**2 + ((z * np.cos(w*z) * w_err) / (np.sin(w*z)))**2) 
            + C_err**2)
        
    def sample(self, z, size=(1,)):
        A = self.sample_A(size)
        w = self.sample_w(size)
        C = self.sample_C(size)
        return self.out(z, A, w, C)

if __name__ == '__main__':
    obj = SineModel()
    prng = np.random.RandomState(123)
    z = prng.uniform(0, 1, 2048)
    x = obj.sample(z)
    
    import matplotlib.pyplot as plt

    plt.scatter(z, x, s=4)
    plt.show()
