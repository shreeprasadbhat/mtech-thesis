import numpy as np
from DarkEnergyModel import DarkEnergyModel

class CPL(DarkEnergyModel) :

    def __init__(self):
        super().__init__()
        self.w0 = -1.9
        self.w0_err = -0.4
        self.wa = -4.0
        self.wa_err = -4.0

    def eos_parameter(self, z): 
        np.seterr(divide='ignore')
        return self.w0 + self.wa * np.divide(z, 1+z)

if __name__ == '__main__':
    obj = CPL()
    obj.distance_modulus(1.)
