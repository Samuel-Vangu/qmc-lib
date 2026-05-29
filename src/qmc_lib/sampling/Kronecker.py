from qmc_lib.core.sampler_core import BaseSampler
from sympy import sieve
import numpy as np

class Kronecker(BaseSampler):

    def __init__(self,dimension,n_samples):
        super().__init__(dimension,n_samples,0)
    
    def generate(self):
        sieve.extend_to_no(self.dim)
        primes_sqrt = np.sqrt(np.array(sieve._list)[0:self.dim])
        tab = np.tile(np.arange(0,self.n_samples), (self.dim,1)).T
        result = np.mod(tab * primes_sqrt,1)
        return result 




