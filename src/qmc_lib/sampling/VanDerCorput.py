from qmc_lib.core.sampler_core import BaseSampler
import numpy as np
from numba import njit

@njit
def generate_VDC(base,n_samples):
    tab = np.arange(0,n_samples)
    result = np.zeros(n_samples)
    facteur = 1/base
    while np.any( tab !=0 )  :
        digit = tab % base
        result += digit * facteur
        tab = tab // base
        facteur = facteur / base
    return result

class VanDerCorput(BaseSampler):
    def __init__(self, base ,n_samples):
        super().__init__(1,n_samples,0)
        self.base = base
    
    def generate(self,backend = "numpy"):
        if backend == "numba":
            return generate_VDC(self.base,self.n_samples)
        tab = np.arange(0,self.n_samples)
        result = np.zeros(self.n_samples)
        facteur = 1/self.base
        while np.any( tab !=0 )  :
            digit = tab % self.base
            result += digit * facteur
            tab = tab // self.base
            facteur = facteur / self.base
        return result
    



 


            


                

