from DarkEnergyModel import DarkEnergyModel

class OmegaCDM(DarkEnergyModel) :

    def __init__(self):
        super().__init__()
        self.w_DE = -1.8
        self.w_DE_err = -0.4

    def eos_parameter(self, z): 
        return self.w_DE  

if __name__ == '__main__':
    obj = OmegaCDM()
    obj.distance_modulus(1.)
