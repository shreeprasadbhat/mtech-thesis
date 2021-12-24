from abc import ABC, abstractmethod
from scipy import integrate
import numpy as np

class DarkEnergyModel(ABC):
    def __init__(self):
        self.omega_M = 0.1
        self.omega_M_err = 0.9
        # Hubble constant, hubble parameter value of present day
        self.H0 = 50
        self.H0_err = 90
        #speed of light in km/s 
        self.c = 299792.458 
        
    @abstractmethod
    def eos_parameter(self, z):
        pass

    def hubble_parameter(self, z): 
        integral, integral_err = integrate.quad(self.eos_parameter, -np.inf, np.inf)
        return self.H0 * (self.omega_M * (1+z)**3 + (1-self.omega_M) * np.exp(3 * integral) )**0.5

    def luminosity_distance(self, z):
        integral, integral_err = integrate.quad(lambda z : 1/self.hubble_parameter(z), 0, z)
        dL =  self.c * (1+z) * integral
        return dL

    def distance_modulus(self, z):
        return 5 * np.log10(self.luminosity_distance(z)) + 25
