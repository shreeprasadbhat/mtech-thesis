from DarkEnergyModel import DarkEnergyModel

class LambdaCDM(DarkEnergyModel) :

    def __init__(self):
        super().__init__()

    def eos_parameter(self, z): 
        return -1  

if __name__ == '__main__':
    obj = LambdaCDM()
    obj.distance_modulus(1.)
