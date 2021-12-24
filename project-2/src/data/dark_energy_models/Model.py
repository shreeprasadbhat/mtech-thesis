from abc import ABC, abstractmethod

class DarkEnergyModel(ABC):
    def __init__(self):
        omega_m0_mean = 0.1
        omega_m0_stddev = 0.9
        # Hubble constant, hubble parameter value of present day
        H_0_mean = 50
        H_0_stddev = 90
        
    @abstractmethod
    def eos(self):
        pass
    def hubble_parameter(selfi, z):
        
    def distance_modulus(self):
    def mu(self)

